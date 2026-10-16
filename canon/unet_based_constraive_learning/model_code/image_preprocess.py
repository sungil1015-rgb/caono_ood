"""Shared image transform for the per-step STFPM models."""

from __future__ import annotations

from io import BytesIO

import torch
from PIL import Image
from torchvision.transforms import functional as TF

ROTATIONS = (0, 90, 180, 270)
TRANSPOSE = {
    0: None,
    90: Image.Transpose.ROTATE_90,
    180: Image.Transpose.ROTATE_180,
    270: Image.Transpose.ROTATE_270,
}


def preprocess(image: Image.Image, size: int, angle: int = 0) -> torch.Tensor:
    """Pad before right-angle rotation, then resize without cropping the photo."""
    if angle not in TRANSPOSE:
        raise ValueError(f"Unsupported rotation: {angle}")
    image = image.convert("RGB")
    side = max(image.size)
    canvas = Image.new("RGB", (side, side), (128, 128, 128))
    canvas.paste(image, ((side - image.width) // 2, (side - image.height) // 2))
    if TRANSPOSE[angle] is not None:
        canvas = canvas.transpose(TRANSPOSE[angle])
    canvas = canvas.resize((size, size), Image.Resampling.LANCZOS)
    tensor = TF.pil_to_tensor(canvas).float().div_(255.0)
    return TF.normalize(tensor, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))


def preprocess_bytes(image_bytes: bytes, size: int, angle: int = 0) -> torch.Tensor:
    with Image.open(BytesIO(image_bytes)) as image:
        return preprocess(image, size, angle)
