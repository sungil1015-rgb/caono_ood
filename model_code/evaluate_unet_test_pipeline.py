"""Run the frozen step classifier -> ROI detector -> U-Net on held-out test images."""
from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms.functional import pil_to_tensor

from evaluate_roi_cross_step_top2 import score_all_steps
from evaluate_roi_unet_contrastive import encode as encode_unet
from image_preprocess import ROTATIONS, preprocess
from known_product_step import build_model, checkpoint_path
from prepare_roi_contrastive import WEIGHTS as DETECTOR_WEIGHTS
from train_roi_step_contrastive import SOURCE, STEPS
from train_roi_unet_contrastive import UNetContrastiveDecoder, load_config, load_encoder
from train_step_roi_detector import make_model
from train_step_contrastive import ROOT
from bundle_paths import ASSET_ROOT, BUNDLE_ROOT, CONFIG_ROOT, OUTPUT_ROOT, unet_model_dir

DATA = ASSET_ROOT / "splits"
OUTPUT = OUTPUT_ROOT / "roi_unet_test_pipeline"
CONFIGS = {
    "3029C003AA": CONFIG_ROOT / "roi_unet_c003_color_only_equal.toml",
    "3029C009AA": CONFIG_ROOT / "roi_unet_contrastive_batch144_epoch20.toml",
}
LEVELS = {"3029C003AA": "up1", "3029C009AA": "bottleneck"}
REVIEWED_NG = {"3029C003AA": ("2EQ16854", 11)}


def product_folder(product: str, config: dict) -> Path:
    return unet_model_dir(product, config)


def rows_for_product(product: str) -> list[dict]:
    with (DATA / "test_truth.csv").open(newline="") as stream:
        test = [dict(row, source_path=row["path"], source_group="test_normal",
                     true_step=int(row["step"])) for row in csv.DictReader(stream)
                if row["product_code"] == product and int(row["step"]) in STEPS]
    ng = []
    for path in sorted((ROOT / "data/canon_data/NG_p").glob(f"{product}__*.jpg")):
        match = re.search(r"step_(\d+)\.jpg$", path.name)
        if not match or int(match.group(1)) not in STEPS:
            continue
        ng.append({"product_code": product, "serial": path.name.split("__")[1],
                   "source_path": str(path.relative_to(ROOT)), "source_group": "ng_folder",
                   "true_step": int(match.group(1))})
    if product in REVIEWED_NG:
        serial, step = REVIEWED_NG[product]
        path = ROOT / "data/canon_data" / product / serial / f"step_{step}.jpg"
        ng.append({"product_code": product, "serial": serial,
                   "source_path": str(path.relative_to(ROOT)), "source_group": "ng_reclassified",
                   "true_step": step})
    expected = {"3029C003AA": 2280, "3029C009AA": 648}[product]
    if len(test) != expected or len({r["source_path"] for r in test + ng}) != len(test + ng):
        raise ValueError("Unexpected test count or duplicate source path")
    if any(not (ROOT / row["source_path"]).is_file() for row in test + ng):
        raise FileNotFoundError("A source image is missing")
    train_or_cal = set()
    for filename in ("train.csv", "val_normal_truth.csv", "val_step_truth.csv"):
        with (DATA / filename).open(newline="") as stream:
            train_or_cal.update(row["path"] for row in csv.DictReader(stream)
                                if row["product_code"] == product)
    if any(row["source_path"] in train_or_cal for row in test):
        raise ValueError("Test image overlaps train or val")
    label_root = ASSET_ROOT / "crop_canon_data"
    detector_labeled = {str(path.parent.relative_to(label_root))
                        for path in label_root.glob("*/*/step_*.txt")}
    if any(f"{product}/{row['serial']}" in detector_labeled for row in test):
        raise ValueError("Test serial overlaps the ROI detector annotation pool")
    return test + ng


class FourViewImages(Dataset):
    def __init__(self, rows: list[dict], size: int):
        self.rows, self.size = rows, size

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        with Image.open(ROOT / self.rows[idx]["source_path"]) as image:
            views = torch.stack([preprocess(image, self.size, angle) for angle in ROTATIONS])
        return views, idx


class DetectorImages(Dataset):
    def __init__(self, rows: list[dict]):
        self.rows = rows

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        with Image.open(ROOT / self.rows[idx]["source_path"]) as image:
            pixels = pil_to_tensor(image.convert("RGB")).float().div_(255)
        return pixels, idx


def detector_collate(batch):
    images, indices = zip(*batch)
    return list(images), list(indices)


@torch.inference_mode()
def classify(rows: list[dict], product: str, device: torch.device, workers: int) -> None:
    checkpoint = torch.load(checkpoint_path(product), map_location="cpu", weights_only=False)
    config = checkpoint["config"]
    model = build_model(int(config["steps"])).to(device).eval()
    model.load_state_dict(checkpoint["model"])
    loader = DataLoader(FourViewImages(rows, int(config["input_size"])), batch_size=16,
                        num_workers=workers, shuffle=False, pin_memory=True)
    for batch_no, (views, indices) in enumerate(loader, 1):
        batch, angles = views.shape[:2]
        logits = model(views.flatten(0, 1).to(device, non_blocking=True)).float()
        logits = logits.reshape(batch, angles, -1).mean(1)
        probabilities = logits.softmax(1)
        predictions = probabilities.argmax(1).tolist()
        confidence = probabilities.max(1).values.tolist()
        for idx, predicted, conf in zip(indices.tolist(), predictions, confidence):
            rows[idx]["predicted_step"] = int(predicted)
            rows[idx]["step_confidence"] = float(conf)
        if batch_no % 30 == 0 or batch_no == len(loader):
            print(product, "classified", min(batch_no * 16, len(rows)), "/", len(rows), flush=True)
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()


def save_roi(row: dict, output: Path, scales: dict[str, float]) -> None:
    if not row["detected"]:
        row.update(crop_path=None, crop_xyxy=None, crop_scale=None)
        return
    with Image.open(ROOT / row["source_path"]) as source:
        image = source.convert("RGB")
        x1, y1, x2, y2 = row["box_xyxy"]
        w, h = x2 - x1, y2 - y1
        box = (max(0, math.floor(x1 - .1 * w)), max(0, math.floor(y1 - .1 * h)),
               min(image.width, math.ceil(x2 + .1 * w)),
               min(image.height, math.ceil(y2 + .1 * h)))
        if box[2] <= box[0] or box[3] <= box[1]:
            row.update(detected=False, crop_path=None, crop_xyxy=None, crop_scale=None)
            return
        crop = image.crop(box)
        # Use the step predicted from pixels, never the test filename's true step.
        candidate_scale = scales.get(str(row["predicted_step"]), 1.0)
        scale = min(candidate_scale, 1024 / max(crop.size))
        size = (max(1, round(crop.width * scale)), max(1, round(crop.height * scale)))
        if crop.size != size:
            crop = crop.resize(size, Image.Resampling.LANCZOS)
        canvas = Image.new("RGB", (1024, 1024), (128, 128, 128))
        canvas.paste(crop, ((1024 - crop.width) // 2, (1024 - crop.height) // 2))
        filename = Path(row["source_path"]).name
        if row["source_group"] != "test_normal":
            filename = f"{row['source_group']}__{row['serial']}__{filename}"
        target = output / "crops" / row["source_group"] / row["serial"] / filename
        target.parent.mkdir(parents=True, exist_ok=True)
        canvas.save(target, quality=95)
    row.update(crop_path=str(target.resolve()), crop_xyxy=list(box), crop_scale=scale,
               crop_scale_step=int(row["predicted_step"]))


@torch.inference_mode()
def detect_and_crop(rows: list[dict], product: str, output: Path,
                    device: torch.device, workers: int) -> None:
    saved = torch.load(DETECTOR_WEIGHTS, map_location="cpu", weights_only=False)
    detector = make_model(len(saved["steps"]) + 1, pretrained=False).to(device).eval()
    detector.load_state_dict(saved["model"])
    roi_manifest = json.loads(SOURCE.read_text())
    scales = roi_manifest["scale_by_step"]
    loader = DataLoader(DetectorImages(rows), batch_size=4, num_workers=workers,
                        shuffle=False, collate_fn=detector_collate, pin_memory=True)
    for batch_no, (images, indices) in enumerate(loader, 1):
        predictions = detector([image.to(device, non_blocking=True) for image in images])
        for idx, prediction in zip(indices, predictions):
            score = float(prediction["scores"][0]) if len(prediction["scores"]) else None
            row = rows[idx]
            row["roi_score"] = score
            row["detected"] = score is not None and score >= 0.25
            row["box_xyxy"] = ([float(x) for x in prediction["boxes"][0].tolist()]
                               if row["detected"] else None)
            save_roi(row, output, scales)
        if batch_no % 50 == 0 or batch_no == len(loader):
            print(product, "detected", min(batch_no * 4, len(rows)), "/", len(rows), flush=True)
            (output / "stage1.json").write_text(json.dumps(rows, ensure_ascii=False) + "\n")
    del detector
    if device.type == "cuda":
        torch.cuda.empty_cache()


@torch.inference_mode()
def score_unet(rows: list[dict], product: str, device: torch.device) -> None:
    config = load_config(CONFIGS[product])
    folder = product_folder(product, config)
    level = LEVELS[product]
    checkpoint = torch.load(folder / "decoder_weights.pt", map_location="cpu", weights_only=False)
    if checkpoint["config"]["hyperparameters"] != config:
        raise ValueError(f"{product} checkpoint and TOML differ")
    encoder = load_encoder(product, device)
    decoder = UNetContrastiveDecoder(config).to(device).eval()
    decoder.load_state_dict(checkpoint["decoder"])
    detected = [row for row in rows if row["detected"]]
    for row in detected:
        row["step"] = row["true_step"]  # RoiWithMask returns a label; scoring checks all candidate steps.
    embedding = encode_unet(encoder, decoder, detected, device, config)[level]
    bank = torch.load(folder / "train_normal_bank.pt", map_location="cpu", weights_only=False)
    reference = bank["embeddings"][level]
    scores = score_all_steps(embedding, bank["rows"], reference, device,
                             int(config["evaluation"]["neighbors"]))
    thresholds = json.loads((folder / "results.json").read_text())["levels"][level]["thresholds"]
    if len(scores) != len(detected):
        raise ValueError("ROI embedding count mismatch")
    for row, score_vector in zip(detected, scores.tolist()):
        row["scores"] = {str(step): float(score_vector[step - 1]) for step in STEPS}
        row["accepted_steps"] = [step for step in STEPS
                                 if row["scores"][str(step)] <= thresholds[str(step)]]
    for row in rows:
        row["unet_expected_pass"] = row["detected"] and row["true_step"] in row["accepted_steps"]
        row["step_matches"] = row["predicted_step"] == row["true_step"]
        row["pipeline_expected_pass"] = row["step_matches"] and row["unet_expected_pass"]
        row["wrong_unet_accepted"] = [step for step in row.get("accepted_steps", [])
                                       if step != row["true_step"]]
        row["wrong_pipeline_accepted"] = [step for step in row["wrong_unet_accepted"]
                                           if step == row["predicted_step"]]
    print(product, "U-Net scored", len(detected), "ROIs", flush=True)


def summarize(rows: list[dict]) -> dict:
    test = [row for row in rows if row["source_group"] == "test_normal"]
    ng = [row for row in rows if row["source_group"] != "test_normal"]
    def basic(group):
        return {"images": len(group),
                "step_correct": sum(row["step_matches"] for row in group),
                "roi_detected": sum(row["detected"] for row in group),
                "unet_expected_pass": sum(row["unet_expected_pass"] for row in group),
                "pipeline_expected_pass": sum(row["pipeline_expected_pass"] for row in group),
                "wrong_attempts": len(group) * (len(STEPS) - 1),
                "wrong_unet_accepted": sum(len(row["wrong_unet_accepted"]) for row in group),
                "wrong_pipeline_accepted": sum(len(row["wrong_pipeline_accepted"]) for row in group),
                "images_with_wrong_unet_accept": sum(bool(row["wrong_unet_accepted"]) for row in group)}
    report = {"test": basic(test), "by_step": {str(step): basic([row for row in test
                                                                  if row["true_step"] == step])
                                                for step in STEPS},
              "ng_folder": basic([row for row in ng if row["source_group"] == "ng_folder"]),
              "ng_reclassified": basic([row for row in ng if row["source_group"] == "ng_reclassified"]),
              "ng_images": [{key: row.get(key) for key in
                            ("source_path", "source_group", "true_step", "predicted_step",
                             "detected", "roi_score", "step_matches", "unet_expected_pass",
                             "pipeline_expected_pass", "wrong_unet_accepted", "accepted_steps")}
                            for row in ng],
              "wrong_unet_matrix": {str(source): {str(target): sum(target in row["wrong_unet_accepted"]
                                                              for row in test if row["true_step"] == source)
                                                  for target in STEPS if target != source}
                                    for source in STEPS},
              "wrong_pipeline_matrix": {str(source): {str(target): sum(target in row["wrong_pipeline_accepted"]
                                                                  for row in test if row["true_step"] == source)
                                                      for target in STEPS if target != source}
                                        for source in STEPS}}
    return report


def write_csv(rows: list[dict], path: Path) -> None:
    columns = ["source_group", "product_code", "serial", "true_step", "source_path",
               "predicted_step", "step_confidence", "step_matches", "detected", "roi_score",
               "crop_path", "unet_expected_pass", "pipeline_expected_pass", "expected_score",
               "expected_threshold", "accepted_steps", "wrong_unet_accepted",
               "wrong_pipeline_accepted"]
    config = load_config(CONFIGS[rows[0]["product_code"]])
    folder = product_folder(rows[0]["product_code"], config)
    level = LEVELS[rows[0]["product_code"]]
    limits = json.loads((folder / "results.json").read_text())["levels"][level]["thresholds"]
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, columns)
        writer.writeheader()
        for row in rows:
            item = {key: row.get(key) for key in columns}
            item["expected_score"] = row.get("scores", {}).get(str(row["true_step"]))
            item["expected_threshold"] = limits[str(row["true_step"])]
            for key in ("accepted_steps", "wrong_unet_accepted", "wrong_pipeline_accepted"):
                item[key] = ";".join(str(step) for step in row.get(key, []))
            writer.writerow(item)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--product", choices=tuple(CONFIGS), required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--reuse-stage1", action="store_true")
    args = parser.parse_args()
    product, device = args.product, torch.device(args.device)
    output = OUTPUT / product
    output.mkdir(parents=True, exist_ok=True)
    expected_rows = rows_for_product(product)
    if args.reuse_stage1 and (output / "stage1.json").is_file():
        rows = json.loads((output / "stage1.json").read_text())
        if [r["source_path"] for r in rows] != [r["source_path"] for r in expected_rows]:
            raise ValueError("Saved stage1 rows do not match current test manifest")
        if any("detected" not in row for row in rows):
            raise ValueError("Saved stage1 is incomplete")
    else:
        rows = expected_rows
        classify(rows, product, device, args.workers)
        detect_and_crop(rows, product, output, device, args.workers)
    score_unet(rows, product, device)
    (output / "per_image.json").write_text(json.dumps(rows, ensure_ascii=False) + "\n")
    write_csv(rows, output / "per_image.csv")
    report = summarize(rows)
    report.update({"product": product, "test_split": "held-out serial-level test; step 1-12",
                   "model_level": LEVELS[product], "model_config": str(CONFIGS[product].relative_to(BUNDLE_ROOT)),
                   "roi_scale_source": "predicted step, not ground-truth file name",
                   "roi_threshold": 0.25,
                   "unet_threshold": "saved per-step val_normal p99",
                   "normal_final_rule": "predicted step == expected step AND ROI detected AND U-Net expected step PASS",
                   "wrong_step_rule": "same image sent to each of the other 11 expected steps"})
    (output / "summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(product, "DONE", report["test"], flush=True)


if __name__ == "__main__":
    main()
