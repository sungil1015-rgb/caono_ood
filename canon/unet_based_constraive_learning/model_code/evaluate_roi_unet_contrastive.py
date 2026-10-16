"""Evaluate frozen-encoder U-Net contrastive embeddings on normal and NG ROIs."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from collections import Counter

import numpy as np
import torch
from torch.utils.data import DataLoader

from train_roi_step_contrastive import SOURCE, STEPS
from bundle_paths import OUTPUT_ROOT, unet_model_dir
from train_roi_unet_contrastive import (
    LEVELS, DEFAULT_CONFIG, OUTPUT, RoiWithMask, UNetContrastiveDecoder, frozen_features, load_encoder,
    load_config, output_root,
)

SPLITS = ('train', 'val_normal', 'val_step', 'ng', 'paired_normal')


@torch.inference_mode()
def encode(encoder, decoder, rows, device, config):
    if not rows:
        return {level: torch.empty(0, config['model']['embedding_dim']) for level in LEVELS}
    settings = config['evaluation']
    batch_size, workers = settings['batch_size'], settings['workers']
    data = DataLoader(RoiWithMask(rows, False, config), batch_size=batch_size, num_workers=workers,
                      pin_memory=False, persistent_workers=workers > 0)
    result = {level: [] for level in LEVELS}
    decoder.eval()
    for batch_no, (images, masks, _) in enumerate(data, 1):
        frozen = frozen_features(encoder, images, device, config, min(batch_size, settings['feature_chunk_size']))
        with torch.autocast('cuda', dtype=torch.bfloat16):
            embeddings = decoder(frozen, masks.to(device, non_blocking=True))
        for level in LEVELS:
            result[level].append(embeddings[level].cpu())
        if batch_no % 50 == 0 or batch_no == len(data):
            print('encoded', min(batch_no * batch_size, len(rows)), '/', len(rows), flush=True)
    return {level: torch.cat(result[level]) for level in LEVELS}


def score_known_step(bank, bank_rows, vectors, rows, device, neighbors):
    if not rows:
        return []
    bank_steps = torch.tensor([int(row['step']) for row in bank_rows], device=device)
    query = torch.as_tensor(vectors, device=device, dtype=torch.float32)
    reference = torch.as_tensor(bank, device=device, dtype=torch.float32)
    scores = np.empty(len(rows), dtype=np.float32)
    query_steps = np.array([int(row['step']) for row in rows])
    for step in STEPS:
        mask = query_steps == step
        if not mask.any():
            continue
        step_bank = reference[bank_steps == step]
        if len(step_bank) < neighbors:
            raise ValueError(f'too few bank examples: step {step}')
        distances = [(1-chunk@step_bank.T).topk(neighbors, dim=1, largest=False).values.mean(dim=1)
                     for chunk in query[mask].split(128)]
        scores[mask] = torch.cat(distances).cpu().numpy()
    return scores


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--product', choices=('3029C003AA', '3029C009AA'), required=True)
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--config', type=Path, default=DEFAULT_CONFIG)
    p.add_argument('--output', type=Path)
    args = p.parse_args()
    config = load_config(args.config)
    device = torch.device(args.device)
    model_folder = unet_model_dir(args.product, config)
    folder = args.output or OUTPUT_ROOT/'unet_evaluation'/model_folder.parent.name/args.product
    folder.mkdir(parents=True, exist_ok=True)
    encoder = load_encoder(args.product, device)
    saved = torch.load(model_folder/'decoder_weights.pt',map_location='cpu',weights_only=False)
    saved_config = saved['config']
    if 'hyperparameters' in saved_config:
        if saved_config['hyperparameters'] != config:
            raise ValueError('Checkpoint hyperparameters differ from supplied TOML')
    elif not (output_root(config).resolve() == OUTPUT.resolve() and
              saved_config['epochs'] == config['train']['epochs'] and
              saved_config['batch_size'] == config['train']['serials_per_step']*len(STEPS)):
        raise ValueError('Legacy checkpoint does not match supplied TOML')
    decoder = UNetContrastiveDecoder(config).to(device).eval()
    decoder.load_state_dict(saved['decoder'])
    manifest = json.loads(SOURCE.read_text())['rows']
    groups = {split:[r for r in manifest if r['product_code']==args.product and r['split']==split]
              for split in SPLITS}
    detected = {split:[r for r in rows if r['detected']] for split,rows in groups.items()}
    vectors = {split:encode(encoder,decoder,detected[split],device,config)
               for split in SPLITS}
    torch.save({'rows':detected['train'],'embeddings':vectors['train']},folder/'train_normal_bank.pt')
    report = {'product':args.product,'config':saved['config'],
              'completed_epochs':saved['epoch'],
              'scoring':f"mean cosine distance to {config['evaluation']['neighbors']} nearest train normal ROI embeddings of the known step",
              'calibration':f"per-step p{100*config['evaluation']['normal_quantile']:g} of detected val_normal; NG and val_step unused",
              'counts':{split:{'total':len(rows),'detected':len(detected[split])}
                        for split,rows in groups.items()},'levels':{}}
    for level in LEVELS:
        scores = {split:score_known_step(vectors['train'][level],detected['train'],
                                         vectors[split][level],detected[split],device,
                                         config['evaluation']['neighbors'])
                  for split in SPLITS[1:]}
        cal_steps = np.array([r['step'] for r in detected['val_normal']])
        thresholds = {str(step):float(np.quantile(np.asarray(scores['val_normal'])[cal_steps==step],config['evaluation']['normal_quantile']))
                      for step in STEPS}
        max_thresholds = {str(step):float(np.max(np.asarray(scores['val_normal'])[cal_steps==step]))
                          for step in STEPS}
        images_by_split, summaries, max_summaries = {}, {}, {}
        for split in SPLITS[1:]:
            lookup = {r['source_path']:float(value) for r,value in zip(detected[split],scores[split])}
            images = []
            for row in groups[split]:
                value = lookup.get(row['source_path'])
                threshold = thresholds[str(row['step'])]
                max_threshold = max_thresholds[str(row['step'])]
                images.append({'path':row['source_path'],'serial':row['serial'],'step':row['step'],
                               'score':value,'threshold':threshold,
                               'score_over_threshold':value/threshold if value is not None else None,
                               'max_score_over_threshold':value/max_threshold if value is not None else None,
                               'decision':'NO_ROI' if value is None else
                                          ('PASS' if value<=threshold else 'FAIL')})
            images_by_split[split] = images
            count = Counter(r['decision'] for r in images)
            summaries[split] = {'total':len(images),'pass':count['PASS'],'fail':count['FAIL'],
                                'no_roi':count['NO_ROI']}
            max_summaries[split] = {
                'total':len(images),
                'pass':sum(r['max_score_over_threshold'] is not None and
                           r['max_score_over_threshold']<=1 for r in images),
                'fail':sum(r['max_score_over_threshold'] is not None and
                           r['max_score_over_threshold']>1 for r in images),
                'no_roi':count['NO_ROI'],
            }
        report['levels'][level] = {'thresholds':thresholds,'max_thresholds':max_thresholds,
                                    'groups':summaries,'max_groups':max_summaries,
                                    'images':images_by_split}
        print(args.product,level,'normal',summaries['val_step'],
              'normal_max',max_summaries['val_step'],
              'NG',summaries['ng'],'paired',summaries['paired_normal'],flush=True)
    (folder/'results.json').write_text(json.dumps(report,indent=2,ensure_ascii=False)+'\n')
    print('saved',folder/'results.json',flush=True)


if __name__=='__main__':
    main()
