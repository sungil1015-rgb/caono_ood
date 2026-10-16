"""Train step-contrastive normal embeddings on cached detector ROIs."""
from __future__ import annotations

import argparse
import json
import math
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset, Sampler
from torchvision.transforms import ColorJitter
from torchvision.transforms import functional as TF

from train_step_contrastive import FourDepthEncoder, LEVELS, ROOT, supervised_contrastive

SOURCE = ROOT / 'data/experiment_v1/roi_contrastive/manifest.json'
STEPS = tuple(range(1, 13))


class RoiImages(Dataset):
    def __init__(self, rows, augment):
        self.rows = rows
        self.augment = augment
        self.jitter = ColorJitter(brightness=.06, contrast=.06, saturation=.04)

    def __len__(self): return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        with Image.open(ROOT / row['crop_path']) as source:
            image = source.convert('RGB')
        if self.augment and random.random() < .7:
            image = self.jitter(image)
        tensor = TF.pil_to_tensor(image).float().div_(255)
        if self.augment:
            tensor = torch.rot90(tensor, random.randrange(4), dims=(-2,-1))
        tensor = TF.normalize(tensor, (.485,.456,.406), (.229,.224,.225))
        return tensor, int(row['step'])


class BalancedStepBatches(Sampler[list[int]]):
    def __init__(self, rows, per_step, seed=1047):
        self.groups = defaultdict(list)
        for index, row in enumerate(rows):
            self.groups[int(row['step'])].append(index)
        if set(self.groups) != set(STEPS):
            raise ValueError('Missing training step after ROI detection')
        self.per_step = per_step
        self.seed = seed
        self.epoch = 0
        self.batches = math.ceil(max(map(len,self.groups.values())) / per_step)
        if any(len(group) < per_step for group in self.groups.values()):
            raise ValueError('Too few positives in a step')

    def set_epoch(self, epoch): self.epoch = epoch
    def __len__(self): return self.batches

    def __iter__(self):
        rng = random.Random(self.seed+self.epoch)
        pools = {step: rng.sample(values, len(values)) for step, values in self.groups.items()}
        offsets = {step: 0 for step in STEPS}
        for _ in range(self.batches):
            indices = []
            for step in STEPS:
                selected = set()
                while len(selected) < self.per_step:
                    if offsets[step] == len(pools[step]):
                        rng.shuffle(pools[step])
                        offsets[step] = 0
                    candidate = pools[step][offsets[step]]
                    offsets[step] += 1
                    selected.add(candidate)
                indices.extend(selected)
            rng.shuffle(indices)
            yield indices


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--product', choices=['3029C003AA','3029C009AA'], required=True)
    p.add_argument('--epochs', type=int, default=8)
    p.add_argument('--serials-per-step', type=int, default=6)
    p.add_argument('--workers', type=int, default=8)
    p.add_argument('--device', default='cuda:1')
    p.add_argument('--output', type=Path)
    args=p.parse_args()
    output=args.output or ROOT/'data/experiment_v1/roi_contrastive'/args.product
    output.mkdir(parents=True,exist_ok=True)
    manifest=json.loads(SOURCE.read_text())
    rows=[r for r in manifest['rows'] if r['product_code']==args.product and r['split']=='train' and r['detected']]
    sampler=BalancedStepBatches(rows,args.serials_per_step)
    random.seed(1047);np.random.seed(1047);torch.manual_seed(1047);torch.cuda.manual_seed_all(1047)
    torch.backends.cudnn.benchmark=True
    loader=DataLoader(RoiImages(rows,True),batch_sampler=sampler,num_workers=args.workers,
                      pin_memory=True,persistent_workers=args.workers>0)
    device=torch.device(args.device)
    model=FourDepthEncoder(pretrained=True).to(device)
    optimizer=torch.optim.AdamW([
        {'params':model.heads.parameters(),'lr':3e-4},
        {'params':model.backbone.layer2.parameters(),'lr':1e-5},
        {'params':model.backbone.layer3.parameters(),'lr':1e-5}],weight_decay=1e-4)
    scaler=torch.amp.GradScaler('cuda')
    config={'product':args.product,'steps':list(STEPS),'input_size':1024,
            'backbone':'ImageNet ResNet18; stem/layer1 frozen, layer2/layer3 fine-tuned',
            'roi_detector':'existing step-class Faster R-CNN, labels ignored at inference',
            'train_images':len(rows),'train_serials':len({r['serial'] for r in rows}),
            'per_step_batch':args.serials_per_step,'batch_size':args.serials_per_step*len(STEPS),
            'epochs':args.epochs,'aug':'0/90/180/270 rotation; mild color jitter',
            'loss':'equal SupCon loss on stem/layer1/layer2/layer3'}
    (output/'config.json').write_text(json.dumps(config,indent=2)+'\n')
    history=[]
    for epoch in range(1,args.epochs+1):
        sampler.set_epoch(epoch)
        model.train()
        losses=[]
        for batch_no,(images,steps) in enumerate(loader,1):
            images=images.to(device,non_blocking=True)
            steps=steps.to(device,non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast('cuda',dtype=torch.float16):
                embeddings=model(images)
                terms=[supervised_contrastive(embeddings[level],steps) for level in LEVELS]
                loss=torch.stack(terms).mean()
            if not torch.isfinite(loss): raise RuntimeError(f'nonfinite loss {epoch}/{batch_no}')
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(),2)
            scaler.step(optimizer);scaler.update()
            losses.append(float(loss.detach()))
            if batch_no%25==0 or batch_no==len(loader):
                print(args.product,'epoch',epoch,'batch',batch_no,'/',len(loader),
                      'loss',round(float(np.mean(losses)),4),flush=True)
        history.append({'epoch':epoch,'mean_loss':float(np.mean(losses))})
        (output/'history.json').write_text(json.dumps(history,indent=2)+'\n')
    torch.save({'model':model.cpu().state_dict(),'config':config},output/'encoder_weights.pt')
    print('saved',output/'encoder_weights.pt',flush=True)


if __name__=='__main__':main()
