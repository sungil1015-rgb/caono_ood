"""Collect held-out images accepted by a U-Net for at least one wrong step."""
from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from bundle_paths import ASSET_ROOT, DATA_ROOT, OUTPUT_ROOT


ROOT = DATA_ROOT
SOURCE = OUTPUT_ROOT / "roi_unet_test_pipeline"
DEST = SOURCE / "unet_wrong_step_pass"
PRODUCTS = ("3029C003AA",)


def make_gallery(rows: list[dict], path: Path, font: ImageFont.FreeTypeFont) -> None:
    width, height, columns = 900, 360, 2
    canvas = Image.new("RGB", (width * columns, height * ((len(rows) + 1) // columns)), "#dedede")
    for index, row in enumerate(rows):
        tile = Image.new("RGB", (width, height), "white")
        draw = ImageDraw.Draw(tile)
        title = (f"{row['product_code']}  {row['serial']}  actual {row['true_step']}  "
                 f"classifier {row['predicted_step']}  U-Net PASS {row['wrong_unet_accepted']}")
        draw.text((12, 8), title, font=font, fill="#b02521")
        draw.text((12, 37), f"final wrong PASS {row['wrong_pipeline_accepted'] or '-'}", font=font, fill="#222222")
        for slot, key in enumerate(("original_copy", "roi_copy")):
            with Image.open(DEST / row[key]) as opened:
                image = opened.convert("RGB")
            if slot == 0 and row["box_xyxy"]:
                ImageDraw.Draw(image).rectangle(row["box_xyxy"], outline="#ec3125", width=5)
            image.thumbnail((428, 280), Image.Resampling.LANCZOS)
            x = slot * 450 + (450 - image.width) // 2
            y = 70 + (280 - image.height) // 2
            tile.paste(image, (x, y))
        canvas.paste(tile, ((index % columns) * width, (index // columns) * height))
    canvas.save(path, quality=90)


def main() -> None:
    DEST.mkdir(parents=True, exist_ok=True)
    font_path = ASSET_ROOT / "fonts/NanumGothic-Bold.ttf"
    font = ImageFont.truetype(str(font_path), 18) if font_path.is_file() else ImageFont.load_default()
    manifest = []
    for product in PRODUCTS:
        rows = json.loads((SOURCE / product / "per_image.json").read_text())
        for row in rows:
            if row["source_group"] != "test_normal" or not row["wrong_unet_accepted"]:
                continue
            assert row["crop_path"] and row["detected"]
            name = f"{row['serial']}__actual_{row['true_step']}__pass_{'-'.join(map(str, row['wrong_unet_accepted']))}.jpg"
            product_dir = DEST / product
            original = product_dir / "original" / name
            roi = product_dir / "roi" / name
            original.parent.mkdir(parents=True, exist_ok=True)
            roi.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(ROOT / row["source_path"], original)
            shutil.copy2(ROOT / row["crop_path"], roi)
            manifest.append({
                "product_code": product,
                "serial": row["serial"],
                "true_step": row["true_step"],
                "predicted_step": row["predicted_step"],
                "wrong_unet_accepted": row["wrong_unet_accepted"],
                "wrong_pipeline_accepted": row["wrong_pipeline_accepted"],
                "source_path": row["source_path"],
                "crop_path": row["crop_path"],
                "original_copy": str(original.relative_to(DEST)),
                "roi_copy": str(roi.relative_to(DEST)),
                "box_xyxy": row["box_xyxy"],
            })
    manifest.sort(key=lambda r: (r["product_code"], r["true_step"], r["serial"]))
    with (DEST / "manifest.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=[k for k in manifest[0] if k != "box_xyxy"])
        writer.writeheader()
        writer.writerows({k: v for k, v in row.items() if k != "box_xyxy"} for row in manifest)
    for product in PRODUCTS:
        rows = [row for row in manifest if row["product_code"] == product]
        for start in range(0, len(rows), 10):
            make_gallery(rows[start:start + 10], DEST / f"{product}_gallery_{start // 10 + 1:02d}.jpg", font)
    wrong_pairs = sum(len(row["wrong_unet_accepted"]) for row in manifest)
    final_pairs = sum(len(row["wrong_pipeline_accepted"]) for row in manifest)
    (DEST / "README.md").write_text(
        "# U-Net 단독 다른 스텝 오수용 이미지\n\n"
        f"테스트 정상 폴더 이미지 **{len(manifest)}장**에서 **{wrong_pairs}건**의 잘못된 스텝 PASS가 있었다. "
        f"이 중 **{final_pairs}건**은 스텝 분류 결과까지 같은 잘못된 스텝이어서 최종 파이프라인도 PASS했다.\n\n"
        "`original/`은 원본 복사본, `roi/`는 검출 뒤 모델에 들어간 ROI 복사본이다. "
        "파일명의 `actual`은 폴더상 실제 스텝, `pass`는 U-Net이 잘못 수용한 스텝이다. "
        "`manifest.csv`의 `predicted_step`은 별도 스텝 분류기 결과다. "
        "갤러리는 원본에 검출 상자를 그려 ROI와 나란히 보여준다.\n\n"
        "정상 폴더의 모든 이미지가 사람에게 확인된 정상이라는 뜻은 아니다. "
        "또한 이 목록은 U-Net 단독 기준이므로, 스텝 분류기가 올바르면 최종 파이프라인에서는 대부분 거절된다.\n",
        encoding="utf-8",
    )
    print(f"images={len(manifest)} wrong_step_pairs={wrong_pairs} final_wrong_pass={final_pairs}")


if __name__ == "__main__":
    main()
