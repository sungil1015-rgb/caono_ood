"""Test two ROI contrastive models against every expected step on val_step."""
from __future__ import annotations

import argparse
import copy
import csv
import json
from pathlib import Path

import torch

from evaluate_roi_resnet18_full_supcon import encode as encode_resnet
from evaluate_roi_unet_contrastive import encode as encode_unet
from train_roi_resnet18_full_supcon import FinalResNetEmbedding, load_config as load_resnet_config
from train_roi_step_contrastive import SOURCE, STEPS
from train_roi_unet_contrastive import (
    UNetContrastiveDecoder,
    load_config as load_unet_config,
    load_encoder,
)
from train_step_contrastive import ROOT

PRODUCT = "3029C003AA"
UNET_CONFIG = Path(__file__).parent / "configs/roi_unet_c003_color_only_equal.toml"
RESNET_CONFIG = Path(__file__).parent / "configs/roi_resnet18_full_supcon_c003_masked.toml"
UNET_FOLDER = ROOT / "data/experiment_v1/roi_unet_contrastive/color_only_equal_batch144_epoch20" / PRODUCT
RESNET_FOLDER = ROOT / "data/experiment_v1/roi_resnet18_full_supcon/c003_masked_mean_max"
DEFAULT_OUTPUT = ROOT / "data/experiment_v1/roi_cross_step_top2_c003"


def load_query_rows() -> list[dict]:
    rows = [row for row in json.loads(SOURCE.read_text())["rows"]
            if row["product_code"] == PRODUCT and row["split"] == "val_step"]
    if not rows or any(not row["detected"] for row in rows):
        raise ValueError("val_step is empty or includes a failed ROI detection")
    if set(int(row["step"]) for row in rows) != set(STEPS):
        raise ValueError("val_step does not cover all trained steps")
    return rows


def get_model_inputs(name: str, rows: list[dict], device: torch.device):
    if name == "unet_up1":
        config = load_unet_config(UNET_CONFIG)
        folder, level = UNET_FOLDER, "up1"
        checkpoint = torch.load(folder / "decoder_weights.pt", map_location="cpu", weights_only=False)
        if checkpoint["config"]["hyperparameters"] != config:
            raise ValueError("U-Net checkpoint and TOML differ")
        encoder = load_encoder(PRODUCT, device)
        decoder = UNetContrastiveDecoder(config).to(device).eval()
        decoder.load_state_dict(checkpoint["decoder"])
        query = encode_unet(encoder, decoder, rows, device, config)[level]
    elif name == "resnet_last":
        config = load_resnet_config(RESNET_CONFIG)
        folder, level = RESNET_FOLDER, "last"
        checkpoint = torch.load(folder / "encoder_weights.pt", map_location="cpu", weights_only=False)
        if checkpoint["config"]["hyperparameters"] != config:
            raise ValueError("ResNet checkpoint and TOML differ")
        model_config = copy.deepcopy(config)
        model_config["model"]["pretrained"] = False
        model = FinalResNetEmbedding(model_config).to(device).eval()
        model.load_state_dict(checkpoint["model"])
        query = encode_resnet(model, rows, device, config)
    else:
        raise ValueError(name)
    result = json.loads((folder / "results.json").read_text())["levels"][level]
    bank = torch.load(folder / "train_normal_bank.pt", map_location="cpu", weights_only=False)
    vectors = bank["embeddings"][level] if name == "unet_up1" else bank["embeddings"]
    return config, query, bank["rows"], vectors, result


@torch.inference_mode()
def score_all_steps(query: torch.Tensor, bank_rows: list[dict], bank: torch.Tensor,
                    device: torch.device, neighbors: int) -> torch.Tensor:
    query = torch.as_tensor(query, device=device, dtype=torch.float32)
    bank = torch.as_tensor(bank, device=device, dtype=torch.float32)
    bank_steps = torch.tensor([int(row["step"]) for row in bank_rows], device=device)
    columns = []
    for step in STEPS:
        reference = bank[bank_steps == step]
        if len(reference) < neighbors:
            raise ValueError(f"Too few reference images for step {step}")
        chunks = [(1 - part @ reference.T).topk(neighbors, dim=1, largest=False).values.mean(1)
                  for part in query.split(128)]
        columns.append(torch.cat(chunks).cpu())
    return torch.stack(columns, dim=1)


def summarize(name: str, rows: list[dict], scores: torch.Tensor, thresholds: dict,
              original: dict) -> tuple[dict, list[dict]]:
    labels = torch.tensor([int(row["step"]) for row in rows], dtype=torch.long)
    limits = torch.tensor([thresholds[str(step)] for step in STEPS], dtype=torch.float64)
    accepted = scores.double() <= limits[None, :]
    own = accepted.gather(1, (labels - 1)[:, None]).squeeze(1)
    wrong = accepted.clone()
    wrong.scatter_(1, (labels - 1)[:, None], False)
    if int(own.sum()) != original["groups"]["val_step"]["pass"]:
        raise ValueError(f"{name}: true-step result differs from saved evaluation")

    matrix = {str(source): {str(target): int(((labels == source) & wrong[:, target - 1]).sum())
                            for target in STEPS if target != source}
              for source in STEPS}
    by_source = {str(source): {"images": int((labels == source).sum()),
                               "wrong_accepted": sum(matrix[str(source)].values())}
                 for source in STEPS}
    by_target = {str(target): {"wrong_attempts": int((labels != target).sum()),
                               "wrong_accepted": sum(matrix[str(source)].get(str(target), 0)
                                                     for source in STEPS if source != target)}
                 for target in STEPS}
    total_wrong = len(rows) * (len(STEPS) - 1)
    wrong_accepted = int(wrong.sum())
    summary = {
        "name": name,
        "query_images": len(rows),
        "true_step_pass": int(own.sum()),
        "true_step_fail": len(rows) - int(own.sum()),
        "wrong_step_attempts": total_wrong,
        "wrong_step_rejected": total_wrong - wrong_accepted,
        "wrong_step_accepted": wrong_accepted,
        "wrong_step_rejection_rate": (total_wrong - wrong_accepted) / total_wrong,
        "images_with_any_wrong_step_accepted": int(wrong.any(dim=1).sum()),
        "images_with_only_true_step_accepted": int((own & ~wrong.any(dim=1)).sum()),
        "by_source_step": by_source,
        "by_expected_step": by_target,
        "wrong_accepted_matrix": matrix,
    }
    details = []
    for idx, row in enumerate(rows):
        details.append({"source_path": row["source_path"], "serial": row["serial"],
                        "true_step": int(row["step"]), "true_step_pass": bool(own[idx]),
                        "wrong_steps_accepted": [step for step in STEPS if wrong[idx, step - 1]]})
    return summary, details


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    device = torch.device(args.device)
    rows = load_query_rows()
    args.output.mkdir(parents=True, exist_ok=True)
    report = {"product": PRODUCT, "split": "val_step", "steps": list(STEPS),
              "design": "Every val_step normal ROI is scored once against each step 1-12. "
                        "Own step should PASS; other steps should FAIL. Each model uses its "
                        "saved per-step p99 val_normal threshold and five-nearest-train cosine score.",
              "models": {}}
    per_image = {row["source_path"]: {"source_path": row["source_path"],
                                       "serial": row["serial"], "true_step": row["step"]}
                 for row in rows}
    for name in ("unet_up1", "resnet_last"):
        print("evaluating", name, flush=True)
        config, query, bank_rows, bank, original = get_model_inputs(name, rows, device)
        scores = score_all_steps(query, bank_rows, bank, device, config["evaluation"]["neighbors"])
        summary, details = summarize(name, rows, scores, original["thresholds"], original)
        report["models"][name] = summary
        for item in details:
            per_image[item["source_path"]][f"{name}_true_pass"] = item["true_step_pass"]
            per_image[item["source_path"]][f"{name}_wrong_pass_steps"] = ",".join(
                str(step) for step in item["wrong_steps_accepted"])
        print(name, "own PASS", summary["true_step_pass"],
              "wrong FAIL", summary["wrong_step_rejected"], "/", summary["wrong_step_attempts"],
              flush=True)
        del query, bank, scores
        if device.type == "cuda":
            torch.cuda.empty_cache()

    (args.output / "summary.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    with (args.output / "per_image.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(next(iter(per_image.values())).keys()))
        writer.writeheader()
        writer.writerows(per_image.values())
    print("saved", args.output, flush=True)


if __name__ == "__main__":
    main()
