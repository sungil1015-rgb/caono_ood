"""Run step classification, ROI detection, and U-Net on one image."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from bundle_paths import OUTPUT_ROOT
from evaluate_unet_test_pipeline import (
    CONFIGS, LEVELS, classify, detect_and_crop, product_folder, score_unet,
)
from train_roi_unet_contrastive import load_config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path)
    parser.add_argument("--product", choices=tuple(CONFIGS), required=True)
    parser.add_argument("--expected-step", type=int, choices=range(1, 13), required=True)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    image = args.image.expanduser().resolve()
    if not image.is_file():
        parser.error(f"Image does not exist: {image}")
    output = args.output or OUTPUT_ROOT / "predict_image" / args.product / image.stem
    output.mkdir(parents=True, exist_ok=True)
    serial = (image.stem.split("__")[1] if "__" in image.stem else
              image.parent.name if image.stem.startswith("step_") else image.stem)
    row = {
        "product_code": args.product,
        "serial": serial,
        "source_path": str(image),
        "source_group": "user_input",
        "true_step": args.expected_step,
    }
    rows = [row]
    device = torch.device(args.device)
    classify(rows, args.product, device, args.workers)
    detect_and_crop(rows, args.product, output, device, args.workers)
    score_unet(rows, args.product, device)
    config = load_config(CONFIGS[args.product])
    limits = json.loads((product_folder(args.product, config) / "results.json").read_text())
    threshold = limits["levels"][LEVELS[args.product]]["thresholds"][str(args.expected_step)]
    fail_reason = (None if row["pipeline_expected_pass"] else
                   "STEP_MISMATCH" if not row["step_matches"] else
                   "NO_ROI" if not row["detected"] else "UNET_FAIL")
    result = {
        "image": str(image),
        "product": args.product,
        "expected_step": args.expected_step,
        "predicted_step": row["predicted_step"],
        "step_confidence": row["step_confidence"],
        "roi_detected": row["detected"],
        "roi_score": row["roi_score"],
        "roi_path": row.get("crop_path"),
        "unet_expected_pass": row["unet_expected_pass"],
        "unet_expected_score": row.get("scores", {}).get(str(args.expected_step)),
        "unet_threshold": threshold,
        "fail_reason": fail_reason,
        "final_decision": "PASS" if row["pipeline_expected_pass"] else "FAIL",
        "accepted_steps_by_unet": row.get("accepted_steps", []),
    }
    (output / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
