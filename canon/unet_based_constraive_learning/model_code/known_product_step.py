"""Recognize a step within a product code supplied by the caller."""

from __future__ import annotations

import sys
from functools import lru_cache
from pathlib import Path

import torch
from torchvision.models import ResNet18_Weights, resnet18

sys.path.insert(0, str(Path(__file__).resolve().parent))
from image_preprocess import ROTATIONS, preprocess_bytes  # noqa: E402
from bundle_paths import MODEL_ROOT

CHECKPOINT_ROOT = MODEL_ROOT / "step_classifier"
DEFAULT_INPUT_SIZE = 384
ACTIVE_INPUT_SIZES = {"3030C001AA": 512}


def build_model(steps: int, pretrained: bool = False) -> torch.nn.Module:
    weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
    model = resnet18(weights=weights)
    model.fc = torch.nn.Linear(model.fc.in_features, steps)
    return model


def checkpoint_path(product_code: str) -> Path:
    size = ACTIVE_INPUT_SIZES.get(product_code, DEFAULT_INPUT_SIZE)
    return CHECKPOINT_ROOT / f"{product_code}_size{size}" / "checkpoint.pt"


@lru_cache(maxsize=8)
def load_model(product_code: str):
    path = checkpoint_path(product_code)
    if not path.is_file():
        raise FileNotFoundError(path)
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    config = checkpoint["config"]
    if config["product_code"] != product_code or config["input_size"] != ACTIVE_INPUT_SIZES.get(product_code, DEFAULT_INPUT_SIZE):
        raise ValueError(f"Checkpoint config does not match {product_code}")
    model = build_model(config["steps"])
    model.load_state_dict(checkpoint["model"])
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()
    return model, device, config


@torch.inference_mode()
def predict_step(image_bytes: bytes, product_code: str) -> dict:
    """Average predictions for the four lossless right-angle rotations."""
    try:
        model, device, config = load_model(product_code)
    except FileNotFoundError:
        return {"step": None, "confidence": None, "decision": "UNAVAILABLE"}
    views = torch.stack([preprocess_bytes(image_bytes, config["input_size"], angle) for angle in ROTATIONS])
    logits = model(views.to(device)).float().mean(dim=0)
    probability = logits.softmax(dim=0)
    step = int(probability.argmax())
    return {"step": step, "confidence": float(probability[step]), "decision": "PREDICTED"}
