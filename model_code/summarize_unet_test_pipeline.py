"""Summarize C003 held-out normals and separately list confirmed C003/C009 NG."""
from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from train_step_contrastive import ROOT
from bundle_paths import ASSET_ROOT, OUTPUT_ROOT, load_ng_reference

BASE = OUTPUT_ROOT / "roi_unet_test_pipeline"
TEST_PRODUCTS = ("3029C003AA",)
NG_PRODUCTS = ("3029C003AA", "3029C009AA")
STEPS = tuple(range(1, 13))


def data():
    reports, images = {}, {}
    for product in NG_PRODUCTS:
        if (BASE / product / "per_image.json").is_file():
            reports[product] = json.loads((BASE / product / "summary.json").read_text())
            images[product] = json.loads((BASE / product / "per_image.json").read_text())
        elif product != TEST_PRODUCTS[0]:
            reports[product], images[product] = load_ng_reference(product)
        else:
            raise FileNotFoundError(BASE / product / "per_image.json")
    return reports, images


def count_stats(rows):
    normal = [r for r in rows if r["source_group"] == "test_normal"]
    reason = Counter()
    for row in normal:
        if row["pipeline_expected_pass"]:
            continue
        if not row["step_matches"]:
            reason["step_mismatch"] += 1
        elif not row["detected"]:
            reason["no_roi"] += 1
        else:
            reason["unet_fail"] += 1
    return {"images": len(normal), "reasons": dict(reason),
            "normal_pipeline_fail": sum(reason.values())}


def save_stage_chart(reports, figure_dir):
    names = ["Step correct", "ROI detected", "U-Net true PASS", "Final PASS"]
    keys = ["step_correct", "roi_detected", "unet_expected_pass", "pipeline_expected_pass"]
    x = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(10, 5.5))
    for index, product in enumerate(TEST_PRODUCTS):
        stats = reports[product]["test"]
        rates = [stats[key] / stats["images"] * 100 for key in keys]
        bars = ax.bar(x, rates, width=.5, label=product)
        for bar, value in zip(bars, rates):
            ax.text(bar.get_x() + bar.get_width()/2, value + .1, f"{value:.2f}%",
                    ha="center", va="bottom", fontsize=9)
    ax.set_xticks(x, names)
    ax.set_ylim(90, 102)
    ax.set_ylabel("Share of held-out normal test images (%)")
    ax.set_title("Classifier → ROI → U-Net: held-out test acceptance")
    ax.legend()
    ax.grid(axis="y", alpha=.2)
    fig.tight_layout()
    fig.savefig(figure_dir / "stage_rates.png", dpi=170)
    plt.close(fig)


def save_step_chart(reports, figure_dir):
    fig, ax = plt.subplots(figsize=(11, 5))
    for product in TEST_PRODUCTS:
        group = reports[product]["by_step"]
        accepted = [group[str(step)]["pipeline_expected_pass"] for step in STEPS]
        total = [group[str(step)]["images"] for step in STEPS]
        ax.plot(STEPS, np.array(accepted)/total*100, marker="o", linewidth=2,
                label=f"{product} ({total[0]} images/step)")
    ax.set_xticks(STEPS)
    ax.set_ylim(75, 101)
    ax.set_xlabel("True step")
    ax.set_ylabel("Final PASS on normal test (%)")
    ax.set_title("Normal acceptance by step")
    ax.grid(alpha=.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(figure_dir / "normal_pass_by_step.png", dpi=170)
    plt.close(fig)


def save_wrong_heatmap(reports, figure_dir):
    fig, ax = plt.subplots(figsize=(8, 6.4), constrained_layout=True)
    axes = [ax]
    all_values = []
    for product in TEST_PRODUCTS:
        matrix = reports[product]["wrong_unet_matrix"]
        all_values.append(np.array([[matrix[str(source)].get(str(target), 0)
                                     for target in STEPS] for source in STEPS]))
    scale = max(1, max(int(value.max()) for value in all_values))
    for ax, product, values in zip(axes, TEST_PRODUCTS, all_values):
        displayed = values.astype(float)
        np.fill_diagonal(displayed, np.nan)
        image = ax.imshow(displayed, cmap="magma", vmin=0, vmax=scale, aspect="equal")
        ax.set_title(f"{product}: wrong-step U-Net PASS counts")
        ax.set_xticks(range(12), STEPS)
        ax.set_yticks(range(12), STEPS)
        ax.set_xlabel("Expected step supplied to U-Net")
        ax.set_ylabel("Image's true step")
        for y in range(12):
            for x in range(12):
                if x != y and values[y, x] > 0:
                    ax.text(x, y, str(values[y, x]), ha="center", va="center",
                            color="white", fontsize=10, fontweight="bold")
    fig.colorbar(image, ax=axes, label="Accepted images (off-diagonal)", shrink=.8)
    fig.savefig(figure_dir / "wrong_step_unet_heatmap.png", dpi=170)
    plt.close(fig)


def save_ng_chart(reports, figure_dir):
    records = [dict(item, product=product) for product in NG_PRODUCTS
               for item in reports[product]["ng_images"]]
    fig, ax = plt.subplots(figsize=(12, 3.7))
    ax.axis("off")
    headers = ["Product / serial", "Expected", "Predicted", "ROI", "U-Net expected", "Final"]
    cells = []
    colors = []
    for item in records:
        serial = Path(item["source_path"]).name.split("__")[1] if item["source_group"] == "ng_folder" else "2EQ16854"
        cells.append([f"{item['product'][-6:]} / {serial}", str(item["true_step"]),
                      str(item["predicted_step"]), "YES" if item["detected"] else "NO",
                      "PASS" if item["unet_expected_pass"] else "FAIL",
                      "PASS" if item["pipeline_expected_pass"] else "FAIL"])
        colors.append("#ffe4e1" if item["pipeline_expected_pass"] else "#e3f2e8")
    table = ax.table(cellText=cells, colLabels=headers, cellLoc="center", loc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2.1)
    for idx, color in enumerate(colors, start=1):
        for col in range(len(headers)):
            table[(idx, col)].set_facecolor(color)
    ax.set_title("Confirmed NG examples: red = missed by full pipeline", pad=14)
    fig.tight_layout()
    fig.savefig(figure_dir / "ng_decisions.png", dpi=170, bbox_inches="tight")
    plt.close(fig)


def save_failure_gallery(images, figure_dir):
    font_path = ASSET_ROOT / "fonts/NanumGothic-Bold.ttf"
    font = ImageFont.truetype(str(font_path), 17) if font_path.is_file() else ImageFont.load_default()
    for product in TEST_PRODUCTS:
        failed = [r for r in images[product]
                  if r["source_group"] == "test_normal" and not r["pipeline_expected_pass"]]
        # First examples per source step show the range of failure modes.
        picked = []
        for step in STEPS:
            picked.extend([r for r in failed if r["true_step"] == step][:2])
        picked = picked[:24]
        if not picked:
            continue
        cols, tile_w, tile_h = 4, 420, 350
        rows = (len(picked) + cols - 1) // cols
        sheet = Image.new("RGB", (cols * tile_w, rows * tile_h), "#e4e4e4")
        for idx, row in enumerate(picked):
            tile = Image.new("RGB", (tile_w, tile_h), "white")
            with Image.open(ROOT / row["source_path"]) as source:
                image = source.convert("RGB")
            if row["box_xyxy"]:
                draw = ImageDraw.Draw(image)
                draw.rectangle(row["box_xyxy"], outline="#e14b30", width=5)
            image.thumbnail((tile_w - 12, tile_h - 72), Image.Resampling.LANCZOS)
            tile.paste(image, ((tile_w - image.width)//2, 70 + (tile_h - 72 - image.height)//2))
            draw = ImageDraw.Draw(tile)
            reason = "STEP" if not row["step_matches"] else "NO ROI" if not row["detected"] else "U-NET"
            draw.text((8, 4), f"{row['serial']} true {row['true_step']} pred {row['predicted_step']}",
                      font=font, fill="#101010")
            draw.text((8, 30), f"FAIL reason: {reason}", font=font, fill="#bc221b")
            sheet.paste(tile, ((idx % cols)*tile_w, (idx // cols)*tile_h))
        sheet.save(figure_dir / f"normal_fail_gallery_{product}.jpg", quality=90)


def main():
    reports, images = data()
    figure_dir = BASE / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    save_stage_chart(reports, figure_dir)
    save_step_chart(reports, figure_dir)
    save_wrong_heatmap(reports, figure_dir)
    save_ng_chart(reports, figure_dir)
    save_failure_gallery(images, figure_dir)
    totals = {key: sum(reports[p]["test"][key] for p in TEST_PRODUCTS)
              for key in reports[TEST_PRODUCTS[0]]["test"]}
    report = {"normal_test_products": {p: {**reports[p]["test"], **count_stats(images[p])}
                                     for p in TEST_PRODUCTS},
              "overall_test": totals,
              "ng_folder": {p: reports[p]["ng_folder"] for p in NG_PRODUCTS},
              "ng_reclassified": {p: reports[p]["ng_reclassified"] for p in NG_PRODUCTS},
              "ng_model_note": "C009 NG was scored with the separately trained C009 model; it is excluded from C003 normal-test metrics.",
              "figures": sorted(str(p.relative_to(BASE)) for p in figure_dir.iterdir() if p.is_file())}
    (BASE / "summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    ng_rows = [row for product in NG_PRODUCTS for row in images[product]
               if row["source_group"] != "test_normal"]
    ng_fields = ("product_code", "serial", "source_group", "source_path", "true_step",
                 "predicted_step", "detected", "unet_expected_pass", "pipeline_expected_pass")
    with (BASE / "ng_examples.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=ng_fields)
        writer.writeheader()
        writer.writerows({field: row[field] for field in ng_fields} for row in ng_rows)
    def line(product):
        stats = reports[product]["test"]
        n = stats["images"]
        wrong = stats["wrong_attempts"]
        return (f"| {product} | {n} | {stats['step_correct']}/{n} | "
                f"{stats['roi_detected']}/{n} | {stats['unet_expected_pass']}/{n} | "
                f"**{stats['pipeline_expected_pass']}/{n}** | "
                f"{wrong-stats['wrong_unet_accepted']}/{wrong} | "
                f"{wrong-stats['wrong_pipeline_accepted']}/{wrong} |")
    n = totals["images"]
    wrong = totals["wrong_attempts"]
    lines = ["# C003 U-Net 전체 파이프라인 보류 test 결과", "",
             "기종을 알고 있는 상태에서 **원본 사진 → ResNet18 스텝 분류(90° 4방향 평균) → "
             "Faster R-CNN 최고 점수 박스 → ROI U-Net** 순서로 평가했다. "
             "정상 수용률과 다른 스텝 거절률은 C003의 스텝 1~12 보류 test 이미지만 사용한다. "
             "ROI crop의 스텝별 크기 설정에는 파일명의 정답 스텝 대신 **분류기가 예측한 스텝**을 사용했다. "
             "학습 가중치·정상 저장소·`val_normal` p99 임계값은 변경하지 않았다.", "",
             "## 1. 정상 스텝 수용과 다른 스텝 거절", "",
             "정상 수용: 예측 스텝이 기대 스텝과 일치하고 ROI가 검출되며, "
             "U-Net이 기대 스텝을 PASS할 때 최종 PASS다. 다른 스텝 평가는 "
             "각 사진을 나머지 11개 기대 스텝에 동일하게 넣는다. "
             "`U-Net 거절`은 점수 단계의 결과이며 ROI가 없으면 FAIL로 포함한다. "
             "`전체 거절`에는 별도 스텝 분류기의 불일치도 포함된다.", "",
             "| 기종 | 정상 사진 | 분류 정답 | ROI 검출 | U-Net 정답 스텝 PASS | 전체 정상 PASS | 다른 스텝 U-Net 거절 | 다른 스텝 전체 거절 |",
             "|---|---:|---:|---:|---:|---:|---:|---:|",
             f"| C003 | {n} | {totals['step_correct']}/{n} | {totals['roi_detected']}/{n} | "
             f"{totals['unet_expected_pass']}/{n} | **{totals['pipeline_expected_pass']}/{n} "
             f"({totals['pipeline_expected_pass']/n:.2%})** | "
             f"{wrong-totals['wrong_unet_accepted']}/{wrong} "
             f"({(wrong-totals['wrong_unet_accepted'])/wrong:.2%}) | "
             f"{wrong-totals['wrong_pipeline_accepted']}/{wrong} "
             f"({(wrong-totals['wrong_pipeline_accepted'])/wrong:.2%}) |", "",
             "다른 스텝의 전체 거절률은 분류기 불일치 조건을 포함하므로 "
             "**U-Net 단독 거절률과 함께** 읽어야 한다. "
             "스텝별 세부 건수와 오수용 조합은 C003 `summary.json`에 있다. "
             "U-Net 단독 다른 스텝 오수용 3건은 [원본·ROI 폴더](unet_wrong_step_pass/README.md)에 모았다.", "",
             "## 2. 정상 FAIL 원인", ""]
    for product in TEST_PRODUCTS:
        reason = count_stats(images[product])["reasons"]
        lines.append(f"- `{product}`: 분류 불일치 {reason.get('step_mismatch',0)}장, "
                     f"분류가 맞았으나 `NO_ROI` {reason.get('no_roi',0)}장, "
                     f"분류·ROI가 맞았으나 U-Net FAIL {reason.get('unet_fail',0)}장.")
    lines += ["", "## 3. NG 시험", "",
              "기존 `NG_p` 확정 사진 4장과 사용자가 정상 목록에서 재확인한 NG 1장을 "
              "**C003 test 정상 분모와 분리**해 평가했다. C009 NG 두 장은 별도로 학습된 C009 모델의 판정이며 C003 모델의 교차 기종 NG 시험이 아니다. "
              "NG의 기대 스텝은 파일명/검토 라벨이다.", "",
              "| 기종 | 시리얼 | NG 출처 | 기대 스텝 | 분류 스텝 | ROI | U-Net 기대 스텝 | 최종 |",
              "|---|---|---|---:|---:|---|---|---|"]
    for product in NG_PRODUCTS:
        for item in reports[product]["ng_images"]:
            serial = Path(item["source_path"]).name.split("__")[1] if item["source_group"] == "ng_folder" else "2EQ16854"
            lines.append(f"| {product} | {serial} | {item['source_group']} | {item['true_step']} | "
                         f"{item['predicted_step']} | {'검출' if item['detected'] else 'NO_ROI'} | "
                         f"{'PASS' if item['unet_expected_pass'] else 'FAIL'} | "
                         f"**{'PASS' if item['pipeline_expected_pass'] else 'FAIL'}** |")
    lines += ["", "NG 표본은 5장뿐이므로 위 FAIL 비율을 일반 결함 검출률로 해석하지 않는다. "
              "`NO_ROI`의 FAIL은 U-Net 이상점수에 의한 검출이 아니다.", "",
              "## 4. 시각화", "",
              "- [단계별 정상 수용률](figures/stage_rates.png)",
              "- [스텝별 최종 정상 PASS](figures/normal_pass_by_step.png)",
              "- [다른 스텝 U-Net 오수용 조합](figures/wrong_step_unet_heatmap.png)",
              "- [NG별 최종 판정](figures/ng_decisions.png)",
              "- [C003 정상 FAIL 41장 전체 갤러리](normal_final_fail/README.md)",
              "- [C003 정상 FAIL 일부 사례](figures/normal_fail_gallery_3029C003AA.jpg)",
              "- [C003의 다른 스텝 U-Net 오수용 원본·ROI](unet_wrong_step_pass/README.md)",
              "- [다른 스텝 최종 오수용 3장](figures/wrong_step_final_pass_gallery.jpg)",
              "- [C003·C009 NG 원본·ROI](figures/ng_image_gallery.jpg)", "",
              "이미지별 원본 경로·스텝 예측·ROI 박스·점수는 "
              "`3029C003AA/per_image.csv`에 있다. NG 별도 사례는 `ng_examples.csv`에 있다. "
              "보류 test 사진은 이번 평가 전 U-Net 학습이나 임계값 선택에 사용하지 않았다.", ""]
    (BASE / "summary.md").write_text("\n".join(lines))
    print("saved", BASE / "summary.md", flush=True)


if __name__ == "__main__":
    main()
