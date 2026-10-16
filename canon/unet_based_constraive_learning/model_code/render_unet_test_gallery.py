"""Show held-out test wrong-step false accepts and NG ROIs for human review."""
from __future__ import annotations

import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from train_step_contrastive import ROOT
from bundle_paths import ASSET_ROOT, OUTPUT_ROOT, load_ng_reference

BASE = OUTPUT_ROOT / 'roi_unet_test_pipeline'
PRODUCTS = ('3029C003AA', '3029C009AA')
NORMAL_TEST_PRODUCT = '3029C003AA'


def sheet(rows: list[dict], destination: Path) -> None:
    font_path = ASSET_ROOT / 'fonts/NanumGothic-Bold.ttf'
    font = ImageFont.truetype(str(font_path), 18) if font_path.is_file() else ImageFont.load_default()
    tile_w, tile_h, columns = 720, 390, 2
    canvas = Image.new('RGB', (tile_w * columns, tile_h * ((len(rows) + 1) // columns)), '#dddddd')
    for index, row in enumerate(rows):
        tile = Image.new('RGB', (tile_w, tile_h), 'white')
        draw = ImageDraw.Draw(tile)
        with Image.open(ROOT / row['source_path']) as source:
            original = source.convert('RGB')
        if row['box_xyxy']:
            ImageDraw.Draw(original).rectangle(row['box_xyxy'], outline='#ef3f2a', width=5)
        original.thumbnail((348, 310), Image.Resampling.LANCZOS)
        tile.paste(original, ((360-original.width)//2, 70+(310-original.height)//2))
        if row['crop_path']:
            with Image.open(ROOT / row['crop_path']) as source:
                roi = source.convert('RGB')
            roi.thumbnail((348, 310), Image.Resampling.LANCZOS)
            tile.paste(roi, (360+(360-roi.width)//2, 70+(310-roi.height)//2))
        draw.text((8, 4), f"{row['product_code']} / {row['serial']}", font=font, fill='#111111')
        draw.text((8, 30), f"true {row['true_step']}  pred {row['predicted_step']}  "
                  f"wrong PASS {row['wrong_pipeline_accepted']}  "
                  f"expected {'PASS' if row['pipeline_expected_pass'] else 'FAIL'}",
                  font=font, fill='#c92221' if row['wrong_pipeline_accepted'] else '#222222')
        canvas.paste(tile, ((index % columns)*tile_w, (index // columns)*tile_h))
    destination.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(destination, quality=90)


def main() -> None:
    images = []
    for product in PRODUCTS:
        path = BASE / product / 'per_image.json'
        images.extend(json.loads(path.read_text()) if path.is_file() else
                      load_ng_reference(product)[1])
    wrong = [row for row in images if row['source_group'] == 'test_normal'
             and row['product_code'] == NORMAL_TEST_PRODUCT
             and row['wrong_pipeline_accepted']]
    ng = [row for row in images if row['source_group'] != 'test_normal']
    outputs = {
        'wrong_step_final_pass_gallery.jpg': wrong,
        'ng_image_gallery.jpg': ng,
    }
    for filename, rows in outputs.items():
        sheet(rows, BASE / 'figures' / filename)
        print(filename, len(rows))
    report_path = BASE / 'summary.json'
    report = json.loads(report_path.read_text())
    report['figures'] = sorted(str(path.relative_to(BASE)) for path in (BASE / 'figures').iterdir())
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')


if __name__ == '__main__':
    main()
