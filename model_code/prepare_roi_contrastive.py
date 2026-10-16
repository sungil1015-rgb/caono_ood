"""Use the existing Faster R-CNN as a class-agnostic box proposer and cache ROI images."""
from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms.functional import pil_to_tensor

from train_step_roi_detector import make_model
from bundle_paths import ASSET_ROOT, BUNDLE_ROOT, DATA_ROOT, MODEL_ROOT, OUTPUT_ROOT

ROOT = DATA_ROOT
DATA = ASSET_ROOT / 'splits'
WEIGHTS = MODEL_ROOT / 'roi_detector/detector_final_weights.pt'
OUTPUT = OUTPUT_ROOT / 'roi_contrastive'
PRODUCTS = ('3029C003AA', '3029C009AA')
STEPS = tuple(range(1, 13))
# These val_step serials were used to train the existing detector.
DETECTOR_TRAIN_OVERLAP = {'3029C003AA/2EQ16834', '3029C003AA/2EQ16859',
                          '3029C003AA/2EQ16872'}


def collect_rows():
    rows = []
    for split, filename in [('train', 'train.csv'), ('val_normal', 'val_normal_truth.csv'),
                            ('val_step', 'val_step_truth.csv')]:
        with (DATA / filename).open(newline='', encoding='utf-8') as stream:
            for row in csv.DictReader(stream):
                if row['product_code'] not in PRODUCTS or int(row['step']) not in STEPS:
                    continue
                if split == 'val_step' and f"{row['product_code']}/{row['serial']}" in DETECTOR_TRAIN_OVERLAP:
                    continue
                rows.append({'split': split, 'product_code': row['product_code'],
                             'serial': row['serial'], 'step': int(row['step']),
                             'source_path': row['path']})
    ng_root = ROOT / 'data/canon_data/NG_p'
    for path in sorted(ng_root.glob('*.jpg')):
        match = re.search(r'step_(\d+)\.jpg$', path.name)
        fields = path.name.split('__')
        if not match or fields[0] not in PRODUCTS or int(match.group(1)) not in STEPS:
            continue
        product, serial, step = fields[0], fields[1], int(match.group(1))
        rows.append({'split': 'ng', 'product_code': product, 'serial': serial,
                     'step': step, 'source_path': str(path.relative_to(ROOT))})
        pair = ROOT / 'data/canon_data' / product / serial / f'step_{step}.jpg'
        if pair.is_file():
            rows.append({'split': 'paired_normal', 'product_code': product,
                         'serial': serial, 'step': step,
                         'source_path': str(pair.relative_to(ROOT))})
    return rows


class SourceImages(Dataset):
    def __init__(self, rows): self.rows = rows
    def __len__(self): return len(self.rows)
    def __getitem__(self, index):
        with Image.open(ROOT / self.rows[index]['source_path']) as image:
            tensor = pil_to_tensor(image.convert('RGB')).float().div_(255)
        return tensor, index


def collate(batch):
    images, indices = zip(*batch)
    return list(images), list(indices)


@torch.inference_mode()
def detect(rows, device, batch_size, workers, min_score):
    saved = torch.load(WEIGHTS, map_location='cpu', weights_only=False)
    model = make_model(len(saved['steps']) + 1, pretrained=False).to(device).eval()
    model.load_state_dict(saved['model'])
    loader = DataLoader(SourceImages(rows), batch_size=batch_size, num_workers=workers,
                        shuffle=False, pin_memory=True, collate_fn=collate,
                        persistent_workers=workers > 0)
    result = [dict(row) for row in rows]
    for batch_no, (images, indices) in enumerate(loader, 1):
        predictions = model([x.to(device, non_blocking=True) for x in images])
        for index, prediction in zip(indices, predictions):
            # Class labels are deliberately ignored. Use only the highest-score box.
            scores = prediction['scores']
            if len(scores) == 0 or float(scores[0]) < min_score:
                result[index].update({'detected': False, 'roi_score': float(scores[0]) if len(scores) else None,
                                      'box_xyxy': None})
            else:
                result[index].update({'detected': True, 'roi_score': float(scores[0]),
                                      'box_xyxy': [float(v) for v in prediction['boxes'][0].tolist()]})
        if batch_no % 100 == 0 or batch_no == len(loader):
            print('detected', min(batch_no * batch_size, len(rows)), '/', len(rows), flush=True)
    return result


def save_crops(rows, output, margin=0.10, side=1024):
    sizes = defaultdict(list)
    for row in rows:
        if row['split'] == 'train' and row['detected']:
            x1, y1, x2, y2 = row['box_xyxy']
            sizes[row['step']].append(max(x2-x1, y2-y1))
    scale = {step: float(np.clip(800 / np.median(sizes[step]), 1, 2)) for step in STEPS}
    for index, row in enumerate(rows, 1):
        if not row['detected']:
            row['crop_path'] = None
            continue
        with Image.open(ROOT / row['source_path']) as image:
            image = image.convert('RGB')
            x1, y1, x2, y2 = row['box_xyxy']
            w, h = x2-x1, y2-y1
            # Expand each side by 10%; clip to original image boundaries.
            box = (max(0, math.floor(x1 - margin*w)), max(0, math.floor(y1 - margin*h)),
                   min(image.width, math.ceil(x2 + margin*w)),
                   min(image.height, math.ceil(y2 + margin*h)))
            if box[2] <= box[0] or box[3] <= box[1]:
                row['detected'] = False
                row['crop_path'] = None
                row['reason'] = 'invalid_box'
                continue
            crop = image.crop(box)
            effective_scale = min(scale[row['step']], side / max(crop.size))
            new_size = (max(1, round(crop.width * effective_scale)),
                        max(1, round(crop.height * effective_scale)))
            if new_size != crop.size:
                crop = crop.resize(new_size, Image.Resampling.LANCZOS)
            canvas = Image.new('RGB', (side, side), (128, 128, 128))
            canvas.paste(crop, ((side-crop.width)//2, (side-crop.height)//2))
            filename = Path(row['source_path']).name
            target = output / 'crops' / row['split'] / row['product_code'] / row['serial'] / filename
            target.parent.mkdir(parents=True, exist_ok=True)
            canvas.save(target, quality=95)
            row['crop_path'] = str(target.resolve())
            row['crop_xyxy'] = box
            row['crop_scale'] = effective_scale
        if index % 1000 == 0 or index == len(rows):
            print('cropped', index, '/', len(rows), flush=True)
    return scale


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=OUTPUT)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--min-score', type=float, default=0.25)
    parser.add_argument('--limit', type=int)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    rows = collect_rows()
    if args.limit: rows = rows[:args.limit]
    print('rows',len(rows),'by split',dict(Counter(row['split'] for row in rows)),flush=True)
    rows = detect(rows, torch.device(args.device), args.batch_size, args.workers, args.min_score)
    print('detection counts',dict(Counter((r['split'],r['product_code'],r['detected']) for r in rows)),flush=True)
    if args.limit:
        (args.output / 'smoke_detection.json').write_text(json.dumps(rows, indent=2)+'\n')
        return
    scale = save_crops(rows, args.output)
    (args.output / 'manifest.json').write_text(json.dumps({
        'detector': str(WEIGHTS.relative_to(BUNDLE_ROOT)),
        'class_use': 'ignore predicted step label; highest-confidence box only',
        'min_score': args.min_score, 'input_canvas': 1024,
        'margin_each_side': 0.10, 'scale_by_step': scale,
        'excluded_detector_train_overlap_val_step_serials': sorted(DETECTOR_TRAIN_OVERLAP),
        'rows': rows}, indent=2, ensure_ascii=False)+'\n')
    print('saved',args.output/'manifest.json',flush=True)


if __name__ == '__main__': main()
