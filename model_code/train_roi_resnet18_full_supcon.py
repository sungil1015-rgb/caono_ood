"""Fine-tune the entire ResNet18 using one final-layer SupCon embedding per ROI."""
from __future__ import annotations

import argparse
import json
import random
import tomllib
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader
from torchvision.models import ResNet18_Weights, resnet18

from train_roi_step_contrastive import BalancedStepBatches, SOURCE, STEPS
from train_roi_unet_contrastive import RoiWithMask, normalize
from train_step_contrastive import ROOT, supervised_contrastive

DEFAULT_CONFIG = Path(__file__).parent / 'configs/roi_resnet18_full_supcon_c003.toml'


def load_config(path: Path) -> dict:
    with path.open('rb') as stream:
        config = tomllib.load(stream)
    train = config['train']
    if config['model']['backbone'] != 'resnet18' or train['scheduler'] != 'cosine':
        raise ValueError('This experiment implements ResNet18 with cosine scheduling')
    if config['model'].get('pooling', 'global_average') not in ('global_average', 'masked_mean_max'):
        raise ValueError('Unknown pooling mode')
    if train['serials_per_step'] < 2 or train['microbatch'] < 1 or train['epochs'] < 1:
        raise ValueError('Invalid training batch or epoch setting')
    if config['augmentation']['rotation_degrees'] != [0]:
        raise ValueError('This comparison requires no rotation augmentation')
    if config['data']['input_size'] % 32:
        raise ValueError('input_size must be divisible by 32')
    return config


class FinalResNetEmbedding(nn.Module):
    def __init__(self, config: dict):
        super().__init__()
        pretrained = config['model']['pretrained']
        self.backbone = resnet18(weights=ResNet18_Weights.DEFAULT if pretrained else None)
        self.backbone.fc = nn.Identity()
        self.pooling = config['model'].get('pooling', 'global_average')
        hidden = config['model']['projection_hidden_dim']
        embedding = config['model']['embedding_dim']
        head_input = 1024 if self.pooling == 'masked_mean_max' else 512
        self.head = nn.Sequential(nn.Linear(head_input, hidden), nn.ReLU(inplace=True),
                                  nn.Linear(hidden, embedding))

    def forward(self, images: torch.Tensor, roi_mask: torch.Tensor | None = None) -> torch.Tensor:
        if self.pooling == 'global_average':
            # Standard ResNet18 layer4 output followed by global average pooling.
            features = self.backbone(images)
        else:
            if roi_mask is None:
                raise ValueError('masked_mean_max requires an ROI mask')
            body = self.backbone
            x = body.relu(body.bn1(body.conv1(images)))
            x = body.layer1(body.maxpool(x))
            x = body.layer2(x)
            x = body.layer3(x)
            x = body.layer4(x)
            valid = F.interpolate(roi_mask.float(), size=x.shape[-2:], mode='nearest') > 0.5
            if not valid.flatten(1).any(dim=1).all():
                raise ValueError('ROI mask vanished at layer4 resolution')
            mean = (x.float()*valid).sum((-2,-1))/valid.sum((-2,-1))
            maximum = x.float().masked_fill(~valid, -torch.inf).amax((-2,-1))
            features = torch.cat((mean, maximum), dim=1)
        return F.normalize(self.head(features).float(), dim=1)


@contextmanager
def no_batchnorm_running_update(model: nn.Module):
    """First embedding pass uses batch statistics without double-updating running stats."""
    modules = [module for module in model.modules() if isinstance(module, nn.modules.batchnorm._BatchNorm)]
    previous = [(module.momentum, module.num_batches_tracked.clone()) for module in modules]
    try:
        for module in modules:
            module.momentum = 0.0
        yield
    finally:
        for module, (momentum, count) in zip(modules, previous):
            module.momentum = momentum
            module.num_batches_tracked.copy_(count)


def train_batch(model, optimizer, images, masks, labels, device, config):
    microbatch = config['train']['microbatch']
    labels = labels.to(device, non_blocking=True)
    # The first pass obtains the embeddings for the exact 144-image loss.
    with torch.no_grad(), no_batchnorm_running_update(model):
        chunks = []
        for part, mask in zip(images.split(microbatch), masks.split(microbatch)):
            with torch.autocast('cuda', dtype=torch.bfloat16):
                chunks.append(model(normalize(part, device, config),
                                    mask.to(device, non_blocking=True)).detach())
    detached = torch.cat(chunks).requires_grad_()
    loss = supervised_contrastive(detached, labels, config['loss']['temperature'])
    if not torch.isfinite(loss):
        raise RuntimeError(f'Nonfinite SupCon loss: {loss}')
    gradient, = torch.autograd.grad(loss, detached)
    optimizer.zero_grad(set_to_none=True)
    # Recompute the same microbatches, backpropagating the full-batch loss gradient.
    for start, (part, mask) in enumerate(zip(images.split(microbatch), masks.split(microbatch))):
        offset = start * microbatch
        with torch.autocast('cuda', dtype=torch.bfloat16):
            embedding = model(normalize(part, device, config), mask.to(device, non_blocking=True))
        (embedding * gradient[offset:offset+len(part)]).sum().backward()
    nn.utils.clip_grad_norm_(model.parameters(), config['train']['grad_clip_norm'])
    optimizer.step()
    return float(loss.detach())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=DEFAULT_CONFIG)
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()
    config = load_config(args.config)
    product = config['experiment']['product']
    device = torch.device(args.device)
    if device.type != 'cuda':
        raise ValueError('Training uses CUDA bfloat16 autocast')
    seed = config['experiment']['seed']
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True
    manifest = json.loads(SOURCE.read_text())['rows']
    rows = [row for row in manifest if row['product_code'] == product and
            row['split'] == 'train' and row['detected']]
    train = config['train']
    sampler = BalancedStepBatches(rows, train['serials_per_step'], seed=seed)
    loader = DataLoader(RoiWithMask(rows, True, config), batch_sampler=sampler,
                        num_workers=train['workers'], pin_memory=False,
                        persistent_workers=train['workers'] > 0)
    model = FinalResNetEmbedding(config).to(device)
    optimizer = torch.optim.AdamW([
        {'params':model.backbone.parameters(), 'lr':train['backbone_learning_rate']},
        {'params':model.head.parameters(), 'lr':train['head_learning_rate']}],
        weight_decay=train['weight_decay'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=train['epochs'])
    folder = ROOT / config['experiment']['output_dir']
    folder.mkdir(parents=True, exist_ok=True)
    metadata = {'product':product, 'steps':list(STEPS), 'train_images':len(rows),
                'batch_size':train['serials_per_step']*len(STEPS),
                'architecture':f"ImageNet ResNet18 layer4 + {config['model'].get('pooling', 'global_average')} + one projection head; full fine-tuning",
                'hyperparameters':config}
    (folder/'config.json').write_text(json.dumps(metadata, indent=2)+'\n')
    (folder/'hyperparameters.toml').write_text(args.config.read_text())
    history = []
    best = float('inf')
    stale = 0
    stop_reason = 'maximum epochs reached'
    for epoch in range(1, train['epochs']+1):
        sampler.set_epoch(epoch)
        model.train()
        losses = []
        for batch_no, (images, masks, labels) in enumerate(loader, 1):
            losses.append(train_batch(model, optimizer, images, masks, labels, device, config))
            if batch_no % 10 == 0 or batch_no == len(loader):
                print(product, 'epoch', epoch, 'batch', batch_no, '/', len(loader),
                      'mean_loss', round(float(np.mean(losses)), 6), flush=True)
        scheduler.step()
        average = float(np.mean(losses))
        if average < best - train['early_stop_min_delta']:
            best, stale = average, 0
        else:
            stale += 1
        history.append({'epoch':epoch, 'mean_loss':average, 'best_mean_loss':best,
                        'stale_epochs':stale})
        (folder/'history.json').write_text(json.dumps(history, indent=2)+'\n')
        saved = {'model':{key:value.detach().cpu() for key,value in model.state_dict().items()},
                 'config':metadata, 'epoch':epoch}
        torch.save(saved, folder/'checkpoint_latest.pt')
        if epoch in train.get('save_epochs', []):
            torch.save(saved, folder/f'encoder_weights_epoch{epoch}.pt')
        print('SAVED', product, 'epoch', epoch, 'loss', round(average, 6), flush=True)
        if epoch >= train['early_stop_min_epochs'] and stale >= train['early_stop_patience']:
            stop_reason = f"loss plateau: min_delta={train['early_stop_min_delta']}, patience={train['early_stop_patience']}"
            print('EARLY_STOP', stop_reason, flush=True)
            break
    (folder/'checkpoint_latest.pt').replace(folder/'encoder_weights.pt')
    metadata['completed_epochs'] = len(history)
    metadata['stop_reason'] = stop_reason
    (folder/'config.json').write_text(json.dumps(metadata, indent=2)+'\n')
    print('COMPLETE', folder/'encoder_weights.pt', flush=True)


if __name__ == '__main__':
    main()
