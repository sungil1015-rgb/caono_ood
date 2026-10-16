"""Train a U-Net decoder on frozen ROI-contrastive ResNet18 features."""
from __future__ import annotations

import argparse
import json
import math
import random
import tomllib
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import ColorJitter
from torchvision.transforms import functional as TF

from train_roi_step_contrastive import BalancedStepBatches, SOURCE, STEPS
from train_step_contrastive import FourDepthEncoder, ROOT, supervised_contrastive
from bundle_paths import BUNDLE_ROOT, CONFIG_ROOT, MODEL_ROOT, OUTPUT_ROOT

OUTPUT = OUTPUT_ROOT / 'roi_unet_contrastive'
DEFAULT_CONFIG = CONFIG_ROOT / 'roi_unet_c003_color_only_equal.toml'
SOURCE_MODELS = MODEL_ROOT / 'roi_encoder'
LEVELS = ('bottleneck', 'up2', 'up1')


def load_config(path: Path) -> dict:
    with path.open('rb') as stream:
        config = tomllib.load(stream)
    train, data, aug, model, loss, evaluation = (config[key] for key in
                                                  ('train', 'data', 'augmentation', 'model', 'loss', 'evaluation'))
    if min(train['epochs'], train['serials_per_step'], train['microbatch'],
           data['input_size'], evaluation['neighbors']) < 1:
        raise ValueError('Batch, epochs, input size, and neighbors must be positive')
    if data['input_size'] % 32:
        raise ValueError('input_size must be divisible by 32')
    if any(deg not in (0, 90, 180, 270) for deg in aug['rotation_degrees']) or not aug['rotation_degrees']:
        raise ValueError('rotation_degrees must contain right-angle rotations')
    if any(model[key] % model['group_norm_groups'] for key in
           ('bottleneck_channels', 'up2_channels', 'up1_channels')):
        raise ValueError('Decoder channels must be divisible by group_norm_groups')
    if sum(loss[f'{level}_weight'] for level in LEVELS) <= 0:
        raise ValueError('At least one loss weight must be positive')
    stop = train.get('stop_after_epochs', train['epochs'])
    if not 1 <= stop <= train['epochs']:
        raise ValueError('stop_after_epochs must be between 1 and epochs')
    if train['scheduler'] != 'cosine':
        raise ValueError('Only cosine scheduler is implemented')
    return config


def output_root(config: dict) -> Path:
    return OUTPUT / Path(config['experiment']['output_dir']).name


class RoiWithMask(Dataset):
    """Return uint8 ROI and mask excluding canvas padding."""

    def __init__(self, rows: list[dict], augment: bool, config: dict):
        self.rows, self.augment, self.config = rows, augment, config
        aug = config['augmentation']
        self.jitter = ColorJitter(brightness=aug['brightness'], contrast=aug['contrast'],
                                  saturation=aug['saturation'])

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        with Image.open(ROOT / row['crop_path']) as source:
            image = source.convert('RGB')
        if self.augment and random.random() < self.config['augmentation']['color_jitter_probability']:
            image = self.jitter(image)
        size = self.config['data']['input_size']
        if image.size != (size, size):
            image = image.resize((size, size), Image.Resampling.BILINEAR)
        pixels = TF.pil_to_tensor(image)
        x1, y1, x2, y2 = row['crop_xyxy']
        width = round((x2-x1)*row['crop_scale'])
        height = round((y2-y1)*row['crop_scale'])
        left, top = (1024-width)//2, (1024-height)//2
        mask_size = size//4
        scale = mask_size/1024
        border = self.config['data']['mask_border_pixels']
        mask = torch.zeros((1, mask_size, mask_size), dtype=torch.uint8)
        mask[:, math.ceil((top+border)*scale):math.floor((top+height-border)*scale),
             math.ceil((left+border)*scale):math.floor((left+width-border)*scale)] = 1
        if not mask.any():
            raise ValueError(f'Empty ROI mask: {row["crop_path"]}')
        rotation = random.choice(self.config['augmentation']['rotation_degrees'])//90 if self.augment else 0
        if rotation:
            pixels = torch.rot90(pixels, rotation, (-2, -1))
            mask = torch.rot90(mask, rotation, (-2, -1))
        return pixels, mask, int(row['step'])


def normalize(images: torch.Tensor, device: torch.device, config: dict) -> torch.Tensor:
    x = images.to(device, non_blocking=True).float().div_(255)
    mean = torch.tensor(config['data']['image_mean'], device=device).view(1, 3, 1, 1)
    std = torch.tensor(config['data']['image_std'], device=device).view(1, 3, 1, 1)
    return (x-mean)/std


def load_encoder(product: str, device: torch.device) -> FourDepthEncoder:
    checkpoint = torch.load(SOURCE_MODELS/product/'encoder_weights.pt',
                            map_location='cpu', weights_only=False)
    encoder = FourDepthEncoder(pretrained=False)
    encoder.load_state_dict(checkpoint['model'])
    encoder.requires_grad_(False)
    return encoder.to(device).eval()


@torch.no_grad()
def frozen_features(encoder, images: torch.Tensor, device: torch.device,
                    config: dict, chunk_size: int) -> dict[str, torch.Tensor]:
    output = {'layer1': [], 'layer2': [], 'layer3': []}
    body = encoder.backbone
    for part in images.split(chunk_size):
        x = normalize(part, device, config)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            x = body.relu(body.bn1(body.conv1(x)))
            l1 = body.layer1(body.maxpool(x))
            l2 = body.layer2(l1)
            l3 = body.layer3(l2)
        for name, feature in zip(output, (l1, l2, l3)):
            output[name].append(feature.detach())
    return {key: torch.cat(parts) for key, parts in output.items()}


class ConvBlock(nn.Sequential):
    def __init__(self, input_channels: int, output_channels: int, groups: int):
        super().__init__(
            nn.Conv2d(input_channels, output_channels, 3, padding=1, bias=False),
            nn.GroupNorm(groups, output_channels), nn.SiLU(inplace=True),
            nn.Conv2d(output_channels, output_channels, 3, padding=1, bias=False),
            nn.GroupNorm(groups, output_channels), nn.SiLU(inplace=True),
        )


class MaskedProjection(nn.Module):
    def __init__(self, channels: int, hidden_dim: int, embedding_dim: int):
        super().__init__()
        self.projection = nn.Sequential(nn.Linear(2*channels, hidden_dim), nn.ReLU(inplace=True),
                                        nn.Linear(hidden_dim, embedding_dim))

    def forward(self, features: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        valid = F.interpolate(mask.float(), size=features.shape[-2:], mode='nearest') > .5
        mean = (features.float()*valid).sum((-2,-1))/valid.sum((-2,-1)).clamp_min(1)
        maximum = features.float().masked_fill(~valid, -torch.inf).amax((-2,-1))
        pooled = torch.cat((mean, maximum), dim=1)
        return F.normalize(self.projection(pooled), dim=1)


class UNetContrastiveDecoder(nn.Module):
    def __init__(self, config: dict):
        super().__init__()
        model = config['model']
        b, u2, u1 = (model['bottleneck_channels'], model['up2_channels'], model['up1_channels'])
        groups = model['group_norm_groups']
        self.bottleneck = ConvBlock(256, b, groups)
        self.up2 = ConvBlock(b+128, u2, groups)
        self.up1 = ConvBlock(u2+64, u1, groups)
        self.heads = nn.ModuleDict({
            'bottleneck': MaskedProjection(b, model['projection_hidden_dim'], model['embedding_dim']),
            'up2': MaskedProjection(u2, model['projection_hidden_dim'], model['embedding_dim']),
            'up1': MaskedProjection(u1, model['projection_hidden_dim'], model['embedding_dim']),
        })

    def forward(self, frozen: dict[str, torch.Tensor], mask: torch.Tensor):
        d3 = self.bottleneck(frozen['layer3'])
        d2 = self.up2(torch.cat((F.interpolate(d3, size=frozen['layer2'].shape[-2:],
                                               mode='bilinear', align_corners=False),
                                 frozen['layer2']), dim=1))
        d1 = self.up1(torch.cat((F.interpolate(d2, size=frozen['layer1'].shape[-2:],
                                               mode='bilinear', align_corners=False),
                                 frozen['layer1']), dim=1))
        return {name: self.heads[name](feature, mask)
                for name, feature in zip(LEVELS, (d3, d2, d1))}


def train_batch(encoder, decoder, optimizer, images, masks, labels, device, config):
    """Exact full-batch SupCon gradient using two decoder passes and small microbatches."""
    microbatch = config['train']['microbatch']
    frozen = frozen_features(encoder, images, device, config, microbatch)
    masks = masks.to(device, non_blocking=True)
    labels = labels.to(device, non_blocking=True)
    batch_size = len(labels)
    with torch.no_grad():
        parts = {level: [] for level in LEVELS}
        for start in range(0, batch_size, microbatch):
            sub = {key: value[start:start+microbatch] for key, value in frozen.items()}
            with torch.autocast('cuda', dtype=torch.bfloat16):
                embeddings = decoder(sub, masks[start:start+microbatch])
            for level in LEVELS:
                parts[level].append(embeddings[level].detach())
    detached = {level: torch.cat(parts[level]).requires_grad_() for level in LEVELS}
    terms = [supervised_contrastive(detached[level], labels,
                                   config['loss']['temperature'])*config['loss'][f'{level}_weight']
             for level in LEVELS]
    loss = torch.stack(terms).sum()/sum(config['loss'][f'{level}_weight'] for level in LEVELS)
    if not torch.isfinite(loss):
        raise RuntimeError(f'Nonfinite SupCon loss: {loss}')
    gradients = torch.autograd.grad(loss, [detached[level] for level in LEVELS])
    optimizer.zero_grad(set_to_none=True)
    for start in range(0, batch_size, microbatch):
        sub = {key: value[start:start+microbatch] for key, value in frozen.items()}
        with torch.autocast('cuda', dtype=torch.bfloat16):
            embeddings = decoder(sub, masks[start:start+microbatch])
        surrogate = sum((embeddings[level]*gradient[start:start+microbatch]).sum()
                        for level, gradient in zip(LEVELS, gradients))
        surrogate.backward()
    nn.utils.clip_grad_norm_(decoder.parameters(), config['train']['grad_clip_norm'])
    optimizer.step()
    return float(loss.detach())


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--product', choices=('3029C003AA','3029C009AA'), required=True)
    p.add_argument('--config', type=Path, default=DEFAULT_CONFIG)
    p.add_argument('--device', default='cuda:0')
    args = p.parse_args()
    config = load_config(args.config)
    train = config['train']
    device = torch.device(args.device)
    seed = config['experiment']['seed']
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True
    manifest = json.loads(SOURCE.read_text())
    rows = [r for r in manifest['rows'] if r['product_code']==args.product and
            r['split']=='train' and r['detected']]
    sampler = BalancedStepBatches(rows, train['serials_per_step'], seed=seed)
    data = DataLoader(RoiWithMask(rows, True, config), batch_sampler=sampler,
                      num_workers=train['workers'], pin_memory=False,
                      persistent_workers=train['workers'] > 0)
    encoder = load_encoder(args.product, device)
    decoder = UNetContrastiveDecoder(config).to(device)
    optimizer = torch.optim.AdamW(decoder.parameters(), lr=train['learning_rate'],
                                  weight_decay=train['weight_decay'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=train['epochs'])
    output = output_root(config)/args.product
    output.mkdir(parents=True, exist_ok=True)
    metadata = {'product':args.product, 'steps':list(STEPS), 'input_size':config['data']['input_size'],
                'frozen_encoder':str((SOURCE_MODELS/args.product/'encoder_weights.pt').relative_to(BUNDLE_ROOT)),
                'train_images':len(rows), 'epochs':train['epochs'],
                'batch_size':train['serials_per_step']*len(STEPS), 'microbatch':train['microbatch'],
                'hyperparameters':config}
    (output/'config.json').write_text(json.dumps(metadata,indent=2)+'\n')
    (output/'hyperparameters.toml').write_text(args.config.read_text())
    history = []
    for epoch in range(1,train.get('stop_after_epochs', train['epochs'])+1):
        sampler.set_epoch(epoch)
        decoder.train()
        losses = []
        for batch_no,(images,masks,labels) in enumerate(data,1):
            value = train_batch(encoder,decoder,optimizer,images,masks,labels,device,config)
            losses.append(value)
            if batch_no%10==0 or batch_no==len(data):
                print(args.product,'epoch',epoch,'batch',batch_no,'/',len(data),
                      'loss',round(float(np.mean(losses)),4),flush=True)
        scheduler.step()
        history.append({'epoch':epoch,'mean_loss':float(np.mean(losses))})
        (output/'history.json').write_text(json.dumps(history,indent=2)+'\n')
        torch.save({'decoder':{key:value.detach().cpu() for key,value in decoder.state_dict().items()},
                    'config':metadata,'epoch':epoch}, output/'checkpoint_latest.pt')
        print('SAVED',args.product,'epoch',epoch,flush=True)
    (output/'checkpoint_latest.pt').replace(output/'decoder_weights.pt')
    metadata['completed_epochs'] = len(history)
    (output/'config.json').write_text(json.dumps(metadata,indent=2)+'\n')


if __name__=='__main__':
    main()
