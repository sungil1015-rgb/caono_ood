"""Four-depth, step-contrastive normal representation for one Canon product.

This trains an OOD feature encoder, not a replacement step classifier.
Product 3029C003AA only; steps 0..12 only. The existing serial split is used.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset, Sampler
from torchvision.models import ResNet18_Weights, resnet18
from torchvision.transforms import ColorJitter

from image_preprocess import ROTATIONS, preprocess

ROOT = Path(__file__).resolve().parents[1]
MANIFESTS = ROOT / "data/experiment_v1"
PRODUCT = "3029C003AA"
STEPS = tuple(range(13))
LEVELS = ("stem", "layer1", "layer2", "layer3")
OUTPUT = MANIFESTS / "contrastive_step_3029C003AA"
INPUT_SIZE = 1024
SEED = 1047


def manifest_rows(filename: str) -> list[dict]:
    with (MANIFESTS / filename).open(newline="", encoding="utf-8") as stream:
        rows = [row for row in csv.DictReader(stream)
                if row["product_code"] == PRODUCT and int(row["step"]) in STEPS]
    for row in rows:
        if not (ROOT / row["path"]).is_file():
            raise FileNotFoundError(ROOT / row["path"])
    return rows


class NormalImages(Dataset):
    def __init__(self, rows: list[dict], augment: bool):
        self.rows = rows
        self.augment = augment
        self.jitter = ColorJitter(brightness=0.06, contrast=0.06, saturation=0.04)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int):
        row = self.rows[index]
        with Image.open(ROOT / row["path"]) as source:
            image = source.convert("RGB")
        if self.augment and random.random() < 0.7:
            image = self.jitter(image)
        angle = random.choice(ROTATIONS) if self.augment else 0
        return preprocess(image, INPUT_SIZE, angle), int(row["step"])


class BalancedSerialBatches(Sampler[list[int]]):
    """Complete devices per batch, giving each step several positive examples."""

    def __init__(self, rows: list[dict], serials_per_batch: int):
        per_serial = defaultdict(dict)
        for index, row in enumerate(rows):
            step = int(row["step"])
            if step in per_serial[row["serial"]]:
                raise ValueError(f"Duplicate image for {row['serial']}/step_{step}")
            per_serial[row["serial"]][step] = index
        self.devices = [per_serial[serial] for serial in sorted(per_serial)]
        if serials_per_batch < 2 or len(self.devices) % serials_per_batch:
            raise ValueError("serials_per_batch must be at least 2 and divide the train serial count")
        self.serials_per_batch = serials_per_batch
        for serial_steps in self.devices:
            if set(serial_steps) != set(STEPS):
                raise ValueError(f"Incomplete device: {serial_steps}")
        self.epoch = 0

    def set_epoch(self, epoch: int):
        self.epoch = epoch

    def __iter__(self):
        indices = list(range(len(self.devices)))
        random.Random(SEED + self.epoch).shuffle(indices)
        for start in range(0, len(indices), self.serials_per_batch):
            batch = []
            for device_index in indices[start:start + self.serials_per_batch]:
                batch.extend(self.devices[device_index][step] for step in STEPS)
            yield batch

    def __len__(self):
        return len(self.devices) // self.serials_per_batch


class ProjectionHead(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.projection = nn.Sequential(nn.Linear(channels * 2, 256), nn.ReLU(inplace=True),
                                        nn.Linear(256, 128))

    def forward(self, features: torch.Tensor):
        pooled = torch.cat((F.adaptive_avg_pool2d(features, 1).flatten(1),
                            F.adaptive_max_pool2d(features, 1).flatten(1)), dim=1)
        return F.normalize(self.projection(pooled).float(), dim=1)


class FourDepthEncoder(nn.Module):
    def __init__(self, pretrained: bool = True):
        super().__init__()
        self.backbone = resnet18(weights=ResNet18_Weights.DEFAULT if pretrained else None)
        self.heads = nn.ModuleDict({
            "stem": ProjectionHead(64), "layer1": ProjectionHead(64),
            "layer2": ProjectionHead(128), "layer3": ProjectionHead(256),
        })
        for parameter in self.backbone.parameters():
            parameter.requires_grad_(False)
        for block in (self.backbone.layer2, self.backbone.layer3):
            for parameter in block.parameters():
                parameter.requires_grad_(True)

    def train(self, mode: bool = True):
        super().train(mode)
        # Preserve the pretrained low-level filters and BatchNorm statistics.
        self.backbone.bn1.eval()
        self.backbone.layer1.eval()
        return self

    def forward(self, image: torch.Tensor):
        body = self.backbone
        stem = body.relu(body.bn1(body.conv1(image)))
        layer1 = body.layer1(body.maxpool(stem))
        layer2 = body.layer2(layer1)
        layer3 = body.layer3(layer2)
        return {name: self.heads[name](feature) for name, feature in
                zip(LEVELS, (stem, layer1, layer2, layer3))}


def supervised_contrastive(embeddings: torch.Tensor, labels: torch.Tensor,
                           temperature: float = 0.1) -> torch.Tensor:
    embeddings = embeddings.float()
    logits = embeddings @ embeddings.T / temperature
    same = labels[:, None] == labels[None, :]
    diag = torch.eye(len(labels), device=labels.device, dtype=torch.bool)
    positives = same & ~diag
    if not positives.any() or not positives.any(dim=1).all():
        raise ValueError("Every anchor must have another image from the same step")
    logits = logits.masked_fill(diag, torch.finfo(logits.dtype).min)
    log_probability = logits - torch.logsumexp(logits, dim=1, keepdim=True)
    return -log_probability[positives].reshape(len(labels), -1).mean()


@torch.inference_mode()
def encode_dataset(model, rows: list[dict], device: torch.device, workers: int) -> tuple[dict, torch.Tensor]:
    dataset = NormalImages(rows, augment=False)
    loader = DataLoader(dataset, batch_size=16, shuffle=False, num_workers=workers,
                        pin_memory=True, persistent_workers=workers > 0)
    model.eval()
    vectors = {name: [] for name in LEVELS}
    labels = []
    for batch_number, (images, steps) in enumerate(loader, 1):
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            output = model(images.to(device, non_blocking=True))
        for name in LEVELS:
            vectors[name].append(output[name].cpu())
        labels.append(steps)
        if batch_number % 80 == 0 or batch_number == len(loader):
            print(f"encoded {min(batch_number * 16, len(rows))}/{len(rows)}", flush=True)
    return {name: torch.cat(parts) for name, parts in vectors.items()}, torch.cat(labels)


def normal_distance_report(train_vectors: dict, train_labels: torch.Tensor,
                           val_vectors: dict, val_labels: torch.Tensor) -> dict:
    device = torch.device("cuda")
    report = {"distance": "mean of 5 nearest train-normal cosine distances for the known step",
              "calibration": "per-step 99th percentile of val_normal scores only", "levels": {}}
    for name in LEVELS:
        bank = train_vectors[name].to(device)
        query = val_vectors[name].to(device)
        labels = train_labels.to(device)
        val_steps = val_labels.to(device)
        records = {}
        margins = []
        for step in STEPS:
            current = query[val_steps == step]
            same_bank = bank[labels == step]
            other_bank = bank[labels != step]
            same_dist = []
            other_dist = []
            for chunk in current.split(128):
                same_dist.append((1 - chunk @ same_bank.T).topk(5, dim=1, largest=False).values.mean(dim=1))
                other_dist.append((1 - chunk @ other_bank.T).min(dim=1).values)
            same_dist = torch.cat(same_dist).cpu().numpy()
            other_dist = torch.cat(other_dist).cpu().numpy()
            margins.extend((other_dist > same_dist).tolist())
            records[str(step)] = {
                "val_images": len(same_dist),
                "same_step_median": float(np.median(same_dist)),
                "other_step_median": float(np.median(other_dist)),
                "threshold_p99_val_normal": float(np.quantile(same_dist, 0.99)),
                "same_step_p95": float(np.quantile(same_dist, 0.95)),
                "same_step_max": float(np.max(same_dist)),
                "other_step_farther_fraction": float(np.mean(other_dist > same_dist)),
            }
        report["levels"][name] = {"by_step": records,
                                  "other_step_farther_total": int(sum(margins)),
                                  "val_images": len(margins)}
        print(name, "other-step farther", sum(margins), "/", len(margins), flush=True)
    return report


def train(args):
    output = args.output_dir or OUTPUT
    output.mkdir(parents=True, exist_ok=True)
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.benchmark = True
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for 1024-pixel training")
    device = torch.device("cuda:0")
    rows = manifest_rows("train.csv")
    val_rows = manifest_rows("val_normal_truth.csv")
    sampler = BalancedSerialBatches(rows, args.serials_per_batch)
    loader = DataLoader(NormalImages(rows, augment=True), batch_sampler=sampler,
                        num_workers=args.workers, pin_memory=True,
                        persistent_workers=args.workers > 0)
    model = FourDepthEncoder(pretrained=not args.resume).to(device)
    optimizer = torch.optim.AdamW([
        {"params": model.heads.parameters(), "lr": 3e-4},
        {"params": model.backbone.layer2.parameters(), "lr": 1e-5},
        {"params": model.backbone.layer3.parameters(), "lr": 1e-5},
    ], weight_decay=1e-4)
    scaler = torch.amp.GradScaler("cuda")
    history = []
    start_epoch = 1
    if args.resume:
        saved = torch.load(args.resume, map_location="cpu", weights_only=False)
        model.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        scaler.load_state_dict(saved["scaler"])
        history = saved["history"]
        start_epoch = saved["epoch"] + 1
    config = {"product_code": PRODUCT, "steps": STEPS, "input_size": INPUT_SIZE,
              "levels": LEVELS,
              "serials_per_batch": args.serials_per_batch,
              "batch_images": args.serials_per_batch * len(STEPS),
              "positives_per_anchor": args.serials_per_batch - 1,
              "loss": "equal mean of four supervised contrastive losses; temperature 0.1",
              "positives": "same step in different train serials",
              "negatives": "other steps of the same product",
              "augmentation": "square pad, random 0/90/180/270 rotation, mild color jitter",
              "backbone": "ImageNet ResNet18; stem+layer1 frozen; layer2+layer3 fine-tuned",
              "train_images": len(rows), "train_serials": len({r["serial"] for r in rows}),
              "val_normal_images": len(val_rows), "excluded_steps": [13, 14]}
    (output / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    for epoch in range(start_epoch, args.epochs + 1):
        sampler.set_epoch(epoch)
        model.train()
        loss_totals = {name: 0.0 for name in LEVELS}
        total = 0.0
        for batch_index, (images, labels) in enumerate(loader, 1):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                embeddings = model(images)
                losses = {name: supervised_contrastive(embeddings[name], labels) for name in LEVELS}
                loss = torch.stack(list(losses.values())).mean()
            if not torch.isfinite(loss):
                raise RuntimeError(f"Nonfinite loss at epoch {epoch} batch {batch_index}")
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            scaler.step(optimizer)
            scaler.update()
            total += float(loss.detach())
            for name in LEVELS:
                loss_totals[name] += float(losses[name].detach())
            if batch_index % 50 == 0 or batch_index == len(loader):
                print(f"epoch {epoch:02d} batch {batch_index:03d}/{len(loader)} loss {total / batch_index:.4f}", flush=True)
            if args.max_batches and batch_index >= args.max_batches:
                break
        count = args.max_batches or len(loader)
        entry = {"epoch": epoch, "mean_loss": total / count,
                 "level_losses": {name: value / count for name, value in loss_totals.items()}}
        history.append(entry)
        checkpoint = {"epoch": epoch, "model": model.state_dict(),
                      "optimizer": optimizer.state_dict(), "scaler": scaler.state_dict(),
                      "history": history, "config": config}
        torch.save(checkpoint, output / "latest_checkpoint.pt")
        (output / "history.json").write_text(json.dumps(history, indent=2) + "\n")
        print("epoch", epoch, json.dumps(entry), flush=True)
    if not args.max_batches:
        torch.save({"model": model.state_dict(), "config": config}, output / "encoder_weights.pt")
        print("encoding train normals for distance bank", flush=True)
        train_vectors, train_labels = encode_dataset(model, rows, device, args.workers)
        torch.save({"embeddings": train_vectors, "labels": train_labels,
                    "product_code": PRODUCT, "steps": STEPS}, output / "train_normal_bank.pt")
        print("encoding val_normal for calibration", flush=True)
        val_vectors, val_labels = encode_dataset(model, val_rows, device, args.workers)
        report = normal_distance_report(train_vectors, train_labels, val_vectors, val_labels)
        (output / "val_normal_distances.json").write_text(json.dumps(report, indent=2) + "\n")
        print("results:", output, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--serials-per-batch", type=int, default=6)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--max-batches", type=int, help="Smoke-test a small number of batches")
    train(parser.parse_args())


if __name__ == "__main__":
    main()
