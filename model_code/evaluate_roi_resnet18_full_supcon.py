"""Evaluate final-layer, fully fine-tuned ResNet18 ROI embeddings for known-step OOD."""
from __future__ import annotations

import argparse
import copy
import json
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from evaluate_roi_unet_contrastive import score_known_step
from train_roi_step_contrastive import SOURCE, STEPS
from train_roi_unet_contrastive import RoiWithMask, normalize
from train_roi_resnet18_full_supcon import DEFAULT_CONFIG, FinalResNetEmbedding, load_config
from train_step_contrastive import ROOT

SPLITS = ('train', 'val_normal', 'val_step', 'ng', 'paired_normal')


@torch.inference_mode()
def encode(model, rows, device, config):
    if not rows:
        return torch.empty(0, config['model']['embedding_dim'])
    settings = config['evaluation']
    batch_size, workers = settings['batch_size'], settings['workers']
    loader = DataLoader(RoiWithMask(rows, False, config), batch_size=batch_size,
                        num_workers=workers, pin_memory=False,
                        persistent_workers=workers > 0)
    chunks = []
    for batch_no, (images, masks, _) in enumerate(loader, 1):
        with torch.autocast('cuda', dtype=torch.bfloat16):
            chunks.append(model(normalize(images, device, config),
                                masks.to(device, non_blocking=True)).cpu())
        if batch_no % 50 == 0 or batch_no == len(loader):
            print('encoded', min(batch_no*batch_size, len(rows)), '/', len(rows), flush=True)
    return torch.cat(chunks)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=DEFAULT_CONFIG)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--suffix', default='')
    args = parser.parse_args()
    config = load_config(args.config)
    product = config['experiment']['product']
    folder = ROOT / config['experiment']['output_dir']
    saved = torch.load(args.checkpoint or folder/'encoder_weights.pt', map_location='cpu', weights_only=False)
    if saved['config']['hyperparameters'] != config:
        raise ValueError('Checkpoint hyperparameters do not match supplied TOML')
    model_config = copy.deepcopy(config)
    model_config['model']['pretrained'] = False
    device = torch.device(args.device)
    model = FinalResNetEmbedding(model_config).to(device).eval()
    model.load_state_dict(saved['model'])
    manifest = json.loads(SOURCE.read_text())['rows']
    groups = {split:[row for row in manifest if row['product_code']==product and
                    row['split']==split] for split in SPLITS}
    detected = {split:[row for row in rows if row['detected']] for split,rows in groups.items()}
    embeddings = {split:encode(model, detected[split], device, config) for split in SPLITS}
    torch.save({'rows':detected['train'], 'embeddings':embeddings['train']},
               folder/f'train_normal_bank{args.suffix}.pt')
    settings = config['evaluation']
    scores = {split:score_known_step(embeddings['train'], detected['train'],
                                    embeddings[split], detected[split], device,
                                    settings['neighbors']) for split in SPLITS[1:]}
    calibration_steps = np.array([row['step'] for row in detected['val_normal']])
    calibration_scores = np.asarray(scores['val_normal'])
    thresholds = {str(step):float(np.quantile(calibration_scores[calibration_steps==step],
                                              settings['normal_quantile'])) for step in STEPS}
    max_thresholds = {str(step):float(np.max(calibration_scores[calibration_steps==step]))
                      for step in STEPS}
    images_by_split, summaries, max_summaries = {}, {}, {}
    for split in SPLITS[1:]:
        lookup = {row['source_path']:float(score) for row,score in
                  zip(detected[split], scores[split])}
        images = []
        for row in groups[split]:
            value = lookup.get(row['source_path'])
            threshold = thresholds[str(row['step'])]
            max_threshold = max_thresholds[str(row['step'])]
            images.append({'path':row['source_path'], 'serial':row['serial'],
                           'step':row['step'], 'score':value, 'threshold':threshold,
                           'score_over_threshold':None if value is None else value/threshold,
                           'max_score_over_threshold':None if value is None else value/max_threshold,
                           'decision':'NO_ROI' if value is None else
                                      ('PASS' if value <= threshold else 'FAIL')})
        images_by_split[split] = images
        counts = Counter(image['decision'] for image in images)
        summaries[split] = {'total':len(images), 'pass':counts['PASS'],
                            'fail':counts['FAIL'], 'no_roi':counts['NO_ROI']}
        max_summaries[split] = {'total':len(images),
                                'pass':sum(image['max_score_over_threshold'] is not None and
                                           image['max_score_over_threshold'] <= 1 for image in images),
                                'fail':sum(image['max_score_over_threshold'] is not None and
                                           image['max_score_over_threshold'] > 1 for image in images),
                                'no_roi':counts['NO_ROI']}
    report = {'product':product, 'config':saved['config'], 'completed_epochs':saved['epoch'],
              'scoring':f"mean cosine distance to {settings['neighbors']} nearest train normal ROI embeddings of the known step",
              'calibration':f"per-step p{100*settings['normal_quantile']:g} of detected val_normal; NG and val_step unused",
              'counts':{split:{'total':len(groups[split]), 'detected':len(detected[split])}
                        for split in SPLITS},
              'levels':{'last':{'thresholds':thresholds, 'max_thresholds':max_thresholds,
                                'groups':summaries, 'max_groups':max_summaries,
                                'images':images_by_split}}}
    result_path = folder/f'results{args.suffix}.json'
    result_path.write_text(json.dumps(report,indent=2,ensure_ascii=False)+'\n')
    print('normal',summaries['val_step'],'normal_max',max_summaries['val_step'],
          'NG',summaries['ng'],'paired',summaries['paired_normal'],flush=True)
    print('saved',result_path,flush=True)


if __name__ == '__main__':
    main()
