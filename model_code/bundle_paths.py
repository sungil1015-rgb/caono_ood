"""Paths for the portable model bundle and an optional external image dataset."""

from __future__ import annotations

import json
import os
from pathlib import Path


BUNDLE_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = Path(os.environ.get("CANON_DATA_ROOT", BUNDLE_ROOT)).expanduser().resolve()
ASSET_ROOT = BUNDLE_ROOT / "assets"
ROI_MANIFEST = Path(os.environ.get("CANON_ROI_MANIFEST", ASSET_ROOT / "roi_manifest.json")).expanduser().resolve()
CONFIG_ROOT = BUNDLE_ROOT / "configs"
MODEL_ROOT = BUNDLE_ROOT / "models"
OUTPUT_ROOT = BUNDLE_ROOT / "outputs"


def data_path(path: str | Path) -> Path:
    """Resolve paths in source manifests, allowing absolute runtime ROI paths."""
    path = Path(path)
    return path if path.is_absolute() else DATA_ROOT / path


def unet_model_dir(product: str, config: dict) -> Path:
    run = Path(config["experiment"]["output_dir"]).name
    return MODEL_ROOT / "unet" / run / product


def comparison_model_dir(config: dict) -> Path:
    run = Path(config["experiment"]["output_dir"]).name
    return MODEL_ROOT / "comparison_resnet" / run


def load_ng_reference(product: str) -> tuple[dict, list[dict]]:
    """Return bundled NG-only examples when a product's batch run is absent."""
    reference = json.loads((ASSET_ROOT / "ng_reference" / f"{product}.json").read_text())
    rows = reference["rows"]
    for row in rows:
        row["source_path"] = str(BUNDLE_ROOT / row["source_path"])
        row["crop_path"] = str(BUNDLE_ROOT / row["crop_path"])
    fields = ("source_path", "source_group", "true_step", "predicted_step",
              "detected", "roi_score", "step_matches", "unet_expected_pass",
              "pipeline_expected_pass", "wrong_unet_accepted", "accepted_steps")
    report = {
        "ng_folder": reference["ng_folder"],
        "ng_reclassified": reference["ng_reclassified"],
        "ng_images": [{key: row.get(key) for key in fields} for row in rows],
    }
    return report, rows
