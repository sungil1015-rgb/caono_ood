"""Collect every C003 held-out normal-folder image rejected by the final pipeline."""
from __future__ import annotations

import csv
import json
import shutil
from collections import Counter
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from bundle_paths import ASSET_ROOT, DATA_ROOT, OUTPUT_ROOT


ROOT = DATA_ROOT
PRODUCT = "3029C003AA"
BASE = OUTPUT_ROOT / "roi_unet_test_pipeline"
SOURCE = BASE / PRODUCT
DEST = BASE / "normal_final_fail"
REASONS = ("STEP_MISMATCH", "NO_ROI", "UNET_FAIL")


def reason(row: dict) -> str:
    if not row["step_matches"]:
        return "STEP_MISMATCH"
    if not row["detected"]:
        return "NO_ROI"
    return "UNET_FAIL"


def gallery(rows: list[dict], target: Path, font: ImageFont.FreeTypeFont) -> None:
    width, height, columns = 900, 370, 2
    canvas = Image.new("RGB", (width * columns, height * ((len(rows) + 1) // columns)), "#dfdfdf")
    for index, row in enumerate(rows):
        tile = Image.new("RGB", (width, height), "white")
        draw = ImageDraw.Draw(tile)
        draw.text((12, 5),
                  f"{row['serial']}  actual {row['true_step']}  pred {row['predicted_step']}  {row['reason']}",
                  font=font, fill="#a12620")
        score = row["expected_score"]
        threshold = row["expected_threshold"]
        detail = (f"U-Net score {score:.5f} / threshold {threshold:.5f}"
                  if score is not None and threshold is not None else "U-Net score unavailable")
        draw.text((12, 33), detail, font=font, fill="#222222")
        with Image.open(DEST / row["original_copy"]) as opened:
            original = opened.convert("RGB")
        if row["box_xyxy"]:
            ImageDraw.Draw(original).rectangle(row["box_xyxy"], outline="#ea3424", width=5)
        original.thumbnail((430, 290), Image.Resampling.LANCZOS)
        tile.paste(original, ((450 - original.width) // 2, 70 + (290 - original.height) // 2))
        if row["roi_copy"]:
            with Image.open(DEST / row["roi_copy"]) as opened:
                roi = opened.convert("RGB")
            roi.thumbnail((430, 290), Image.Resampling.LANCZOS)
            tile.paste(roi, (450 + (450 - roi.width) // 2, 70 + (290 - roi.height) // 2))
        else:
            draw.text((555, 200), "NO ROI", font=font, fill="#a12620")
        canvas.paste(tile, ((index % columns) * width, (index // columns) * height))
    canvas.save(target, quality=90)


def main() -> None:
    rows = json.loads((SOURCE / "per_image.json").read_text())
    scores = {r["source_path"]: r for r in csv.DictReader((SOURCE / "per_image.csv").open())}
    failed = [r for r in rows if r["source_group"] == "test_normal" and not r["pipeline_expected_pass"]]
    if DEST.exists():
        shutil.rmtree(DEST)
    DEST.mkdir(parents=True)
    records = []
    for row in failed:
        kind = reason(row)
        name = f"{row['serial']}__actual_{row['true_step']}__pred_{row['predicted_step']}.jpg"
        original = DEST / kind / "original" / name
        original.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / row["source_path"], original)
        roi_copy = ""
        if row["crop_path"]:
            roi = DEST / kind / "roi" / name
            roi.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(ROOT / row["crop_path"], roi)
            roi_copy = str(roi.relative_to(DEST))
        score_row = scores[row["source_path"]]
        records.append({
            "product_code": PRODUCT,
            "serial": row["serial"],
            "true_step": row["true_step"],
            "predicted_step": row["predicted_step"],
            "reason": kind,
            "detected": row["detected"],
            "unet_expected_pass": row["unet_expected_pass"],
            "expected_score": float(score_row["expected_score"]) if score_row["expected_score"] else None,
            "expected_threshold": float(score_row["expected_threshold"]) if score_row["expected_threshold"] else None,
            "source_path": row["source_path"],
            "crop_path": row["crop_path"] or "",
            "original_copy": str(original.relative_to(DEST)),
            "roi_copy": roi_copy,
            "box_xyxy": row["box_xyxy"],
        })
    records.sort(key=lambda r: (REASONS.index(r["reason"]), r["true_step"], r["serial"]))
    with (DEST / "manifest.csv").open("w", newline="") as stream:
        fields = [key for key in records[0] if key != "box_xyxy"]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows({key: row[key] for key in fields} for row in records)
    font_path = ASSET_ROOT / "fonts/NanumGothic-Bold.ttf"
    font = ImageFont.truetype(str(font_path), 18) if font_path.is_file() else ImageFont.load_default()
    links = []
    for kind in REASONS:
        group = [r for r in records if r["reason"] == kind]
        for start in range(0, len(group), 8):
            name = f"{kind.lower()}_gallery_{start // 8 + 1:02d}.jpg"
            gallery(group[start:start + 8], DEST / name, font)
            links.append(f"- [{kind} {start + 1}–{min(start + 8, len(group))}](./{name})")
    counts = Counter(r["reason"] for r in records)
    (DEST / "README.md").write_text(
        "# C003 정상 폴더 이미지의 최종 FAIL 41장\n\n"
        f"보류 test 정상 폴더에서 최종 FAIL된 {len(records)}장을 전부 모았다. "
        f"스텝 분류 불일치 {counts['STEP_MISMATCH']}장, ROI 미검출 {counts['NO_ROI']}장, "
        f"분류와 ROI가 성공했지만 U-Net이 거절한 이미지 {counts['UNET_FAIL']}장이다. "
        "이 분류는 최종 FAIL의 첫 원인을 기준으로 한다.\n\n"
        "갤러리 왼쪽은 원본 사진이며 빨간 사각형은 검출된 ROI 박스다. 오른쪽은 모델에 넣은 ROI다. "
        "파일명과 `manifest.csv`에는 실제 스텝, 예측 스텝, U-Net 점수와 임계값이 담겨 있다. "
        "`NO_ROI`는 ROI와 U-Net 점수가 없다. 정상 폴더의 라벨은 전체 이미지를 사람이 다시 확인한 결과는 아니다.\n\n"
        "## 원인별 갤러리\n\n" + "\n".join(links) + "\n",
        encoding="utf-8",
    )
    print(f"images={len(records)} reasons={dict(counts)} galleries={len(links)}")


if __name__ == "__main__":
    main()
