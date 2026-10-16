"""Train one pretrained Faster R-CNN detector with step as the box class.

YOLO txt labels are read from crop_canon_data, while pixels are read from
canon_data. Five complete serial folders are held out and visualized every
five epochs. No product code is passed to the detector.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import tv_tensors
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_V2_Weights, fasterrcnn_resnet50_fpn_v2
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.ops import box_iou
from torchvision.transforms import v2

from bundle_paths import ASSET_ROOT, DATA_ROOT, OUTPUT_ROOT

ROOT = DATA_ROOT
LABEL_ROOT = ASSET_ROOT / "crop_canon_data"
IMAGE_ROOT = DATA_ROOT / "data/canon_data"
DEFAULT_OUTPUT = OUTPUT_ROOT / "roi_detector"
HOLDOUT = (
    "3029C003AA/2EQ16174",
    "3029C003AA/2EQ16206",
    "3029C003AA/2EQ16310",
    "3029C003AA/2EQ16873",
    "3029C009AA/2EX36399",
)


def collect() -> tuple[list[dict], list[int]]:
    records = []
    for label_path in sorted(LABEL_ROOT.glob("*/*/step_*.txt")):
        rel = label_path.relative_to(LABEL_ROOT)
        step = int(label_path.stem.removeprefix("step_"))
        image_path = IMAGE_ROOT / rel.with_suffix(".jpg")
        if not image_path.is_file():
            raise FileNotFoundError(image_path)
        raw = []
        for line in label_path.read_text().splitlines():
            parts = line.split()
            if len(parts) != 5 or int(parts[0]) != 0:
                raise ValueError(f"Unexpected YOLO label: {label_path}: {line}")
            cx, cy, bw, bh = map(float, parts[1:])
            if not all(0 <= x <= 1 for x in (cx, cy, bw, bh)) or bw <= 0 or bh <= 0:
                raise ValueError(f"Invalid box: {label_path}: {line}")
            raw.append((cx, cy, bw, bh))
        if not raw:
            raise ValueError(f"No box: {label_path}")
        records.append({"image_path": image_path, "label_path": label_path,
                        "serial_folder": str(rel.parent), "step": step, "yolo_boxes": raw})
    steps = sorted({r["step"] for r in records})
    if not records or not all({*range(1, 13)} <= {r["step"] for r in records if r["serial_folder"] == folder}
                              for folder in HOLDOUT):
        raise ValueError("A held-out folder is missing steps 1..12")
    return records, steps


class StepRoiDataset(Dataset):
    def __init__(self, records: list[dict], step_to_label: dict[int, int], augment: bool):
        self.records = records
        self.step_to_label = step_to_label
        self.transform = v2.Compose([
            v2.ToImage(),
            *([v2.RandomAffine(degrees=3, translate=(0.02, 0.02), fill=128),
                v2.ColorJitter(brightness=0.08, contrast=0.08, saturation=0.06)] if augment else []),
            v2.ToDtype(torch.float32, scale=True),
            v2.ToPureTensor(),
        ])

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int):
        record = self.records[index]
        with Image.open(record["image_path"]) as source:
            image = source.convert("RGB")
        width, height = image.size
        boxes = []
        for cx, cy, bw, bh in record["yolo_boxes"]:
            boxes.append([(cx - bw / 2) * width, (cy - bh / 2) * height,
                          (cx + bw / 2) * width, (cy + bh / 2) * height])
        target = {
            "boxes": tv_tensors.BoundingBoxes(boxes, format="XYXY", canvas_size=(height, width), dtype=torch.float32),
            "labels": torch.tensor([self.step_to_label[record["step"]]] * len(boxes), dtype=torch.int64),
            "image_id": torch.tensor(index),
        }
        image, target = self.transform(image, target)
        target["boxes"] = target["boxes"].clamp(min=0)
        return image, target


def collate(batch):
    images, targets = zip(*batch)
    return list(images), list(targets)


def make_model(num_classes: int, pretrained: bool = True):
    weights = FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT if pretrained else None
    model = fasterrcnn_resnet50_fpn_v2(
        weights=weights, weights_backbone=None, min_size=768, max_size=1024,
        box_detections_per_img=10,
    )
    model.roi_heads.box_predictor = FastRCNNPredictor(
        model.roi_heads.box_predictor.cls_score.in_features, num_classes)
    return model


@torch.inference_mode()
def predict_records(model, dataset, device, batch_size):
    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=2, collate_fn=collate)
    results = []
    for images, targets in loader:
        predictions = model([image.to(device) for image in images])
        for prediction, target in zip(predictions, targets):
            results.append((
                {k: v.detach().cpu() for k, v in prediction.items()},
                {k: v.detach().cpu() for k, v in target.items()},
            ))
    return results


def evaluate(predictions, threshold=0.25):
    hits = 0
    ious = []
    by_step = {}
    for (prediction, target) in predictions:
        step_label = int(target["labels"][0])
        eligible = (prediction["labels"] == step_label) & (prediction["scores"] >= threshold)
        best_iou = 0.0
        if eligible.any():
            overlap = box_iou(prediction["boxes"][eligible], target["boxes"])
            best_iou = float(overlap.max())
        hit = best_iou >= 0.5
        hits += hit
        ious.append(best_iou)
        stat = by_step.setdefault(step_label, {"hits": 0, "total": 0})
        stat["hits"] += hit
        stat["total"] += 1
    return {"iou50_recall_at_score_0.25": hits / len(predictions),
            "hits": hits, "images": len(predictions), "mean_best_correct_class_iou": float(np.mean(ious)),
            "by_label": by_step}


def draw_sample(record, prediction, target, label_to_step, font):
    with Image.open(record["image_path"]) as source:
        image = source.convert("RGB")
    draw = ImageDraw.Draw(image)
    for box in target["boxes"].tolist():
        draw.rectangle(box, outline="#15d65c", width=5)
    shown = 0
    for box, label, score in zip(prediction["boxes"].tolist(), prediction["labels"].tolist(), prediction["scores"].tolist()):
        if score < 0.10 or shown >= 3:
            break
        color = "#f3303b" if label_to_step[label] != record["step"] else "#0072e5"
        draw.rectangle(box, outline=color, width=4)
        draw.text((max(0, box[0]), max(0, box[1] - 22)), f"pred {label_to_step[label]} {score:.2f}",
                  fill=color, stroke_width=2, stroke_fill="white", font=font)
        shown += 1
    if shown == 0:
        draw.text((10, 10), "no pred >= 0.10", fill="red", stroke_width=2, stroke_fill="white", font=font)
    return image


def visualize(records, predictions, label_to_step, output, epoch):
    font_path = Path(__file__).with_name("assets") / "fonts/NanumGothic-Bold.ttf"
    font = ImageFont.truetype(str(font_path), 20) if font_path.is_file() else ImageFont.load_default()
    epoch_dir = output / "visualizations" / f"epoch_{epoch:03d}"
    epoch_dir.mkdir(parents=True, exist_ok=True)
    tile_w, tile_h = 512, 384
    for folder in HOLDOUT:
        indices = [i for i, record in enumerate(records) if record["serial_folder"] == folder and record["step"] in range(1, 13)]
        indices.sort(key=lambda i: records[i]["step"])
        assert len(indices) == 12
        sheet = Image.new("RGB", (tile_w * 4, (tile_h + 34) * 3), "#eeeeee")
        sheet_draw = ImageDraw.Draw(sheet)
        for cell, index in enumerate(indices):
            record = records[index]
            prediction, target = predictions[index]
            image = draw_sample(record, prediction, target, label_to_step, font)
            image.thumbnail((tile_w, tile_h), Image.Resampling.LANCZOS)
            col, row = cell % 4, cell // 4
            x, y = col * tile_w, row * (tile_h + 34)
            sheet.paste(image, (x, y + 34))
            sheet_draw.text((x + 8, y + 5), f"step {record['step']}  GT green / correct blue / wrong red",
                            fill="black", font=font)
        target_path = epoch_dir / f"{folder.replace('/', '__')}.jpg"
        sheet.save(target_path, quality=92)
    return epoch_dir


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--resume", type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    records, steps = collect()
    step_to_label = {step: i + 1 for i, step in enumerate(steps)}
    label_to_step = {label: step for step, label in step_to_label.items()}
    train_records = [r for r in records if r["serial_folder"] not in HOLDOUT]
    val_records = [r for r in records if r["serial_folder"] in HOLDOUT]
    manifest = {"model": "torchvision Faster R-CNN ResNet50 FPN v2, COCO pretrained",
                "classes": steps, "step_to_label": step_to_label,
                "augment": "RandomAffine +/-3 degrees, translate 2%; ColorJitter brightness/contrast .08 saturation .06",
                "holdout_serial_folders": HOLDOUT, "train_images": len(train_records), "holdout_images": len(val_records),
                "train_by_step": dict(sorted(Counter(r["step"] for r in train_records).items())),
                "holdout_by_step": dict(sorted(Counter(r["step"] for r in val_records).items())),
                "visualization": "Every 5 epochs, five 1..12 step contact sheets. GT green, correct class blue, wrong class red."}
    (args.output / "split.json").write_text(json.dumps(manifest, indent=2) + "\n")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("This training recipe requires CUDA")
    print(json.dumps({"device": str(device), "train": len(train_records), "holdout": len(val_records),
                      "classes": steps, "holdout_folders": HOLDOUT}), flush=True)
    train_set = StepRoiDataset(train_records, step_to_label, augment=True)
    val_set = StepRoiDataset(val_records, step_to_label, augment=False)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.workers, collate_fn=collate, pin_memory=True)
    model = make_model(len(steps) + 1, pretrained=args.resume is None).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0001, weight_decay=0.0001)
    scaler = torch.amp.GradScaler("cuda")
    start_epoch = 1
    history = []
    if args.resume:
        saved = torch.load(args.resume, map_location="cpu", weights_only=False)
        model.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        scaler.load_state_dict(saved["scaler"])
        start_epoch = saved["epoch"] + 1
        history = saved["history"]
    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        total_loss = 0.0
        for batch_index, (images, targets) in enumerate(train_loader, 1):
            images = [image.to(device, non_blocking=True) for image in images]
            targets = [{key: value.to(device, non_blocking=True) for key, value in target.items()}
                       for target in targets]
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda"):
                loss_dict = model(images, targets)
                loss = sum(loss_dict.values())
            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite loss in epoch {epoch}, batch {batch_index}: {loss_dict}")
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_loss += float(loss.detach())
            if batch_index % 15 == 0:
                print(f"epoch {epoch:02d} batch {batch_index:02d}/{len(train_loader)} loss {float(loss.detach()):.4f}", flush=True)
        entry = {"epoch": epoch, "train_loss": total_loss / len(train_loader)}
        if epoch % 5 == 0 or epoch == args.epochs:
            predictions = predict_records(model, val_set, device, args.batch_size)
            metrics = evaluate(predictions)
            entry["holdout"] = metrics
            epoch_dir = visualize(val_records, predictions, label_to_step, args.output, epoch)
            checkpoint = {"epoch": epoch, "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                          "scaler": scaler.state_dict(), "steps": steps, "history": history + [entry]}
            torch.save(checkpoint, args.output / f"checkpoint_epoch_{epoch:03d}.pt")
            if epoch == args.epochs:
                torch.save({"model": model.state_dict(), "steps": steps, "epoch": epoch},
                           args.output / "detector_final_weights.pt")
            print(f"epoch {epoch:02d} loss {entry['train_loss']:.4f} recall50 {metrics['hits']}/{metrics['images']} "
                  f"visualizations {epoch_dir}", flush=True)
        else:
            print(f"epoch {epoch:02d} loss {entry['train_loss']:.4f}", flush=True)
        history.append(entry)
        (args.output / "history.json").write_text(json.dumps(history, indent=2) + "\n")


if __name__ == "__main__":
    main()
