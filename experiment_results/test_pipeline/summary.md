# C003 U-Net 전체 파이프라인 보류 test 결과

기종을 알고 있는 상태에서 **원본 사진 → ResNet18 스텝 분류(90° 4방향 평균) → Faster R-CNN 최고 점수 박스 → ROI U-Net** 순서로 평가했다. 정상 수용률과 다른 스텝 거절률은 C003의 스텝 1~12 보류 test 이미지만 사용한다. ROI crop의 스텝별 크기 설정에는 파일명의 정답 스텝 대신 **분류기가 예측한 스텝**을 사용했다. 학습 가중치·정상 저장소·`val_normal` p99 임계값은 변경하지 않았다.

## 1. 정상 스텝 수용과 다른 스텝 거절

정상 수용: 예측 스텝이 기대 스텝과 일치하고 ROI가 검출되며, U-Net이 기대 스텝을 PASS할 때 최종 PASS다. 다른 스텝 평가는 각 사진을 나머지 11개 기대 스텝에 동일하게 넣는다. `U-Net 거절`은 점수 단계의 결과이며 ROI가 없으면 FAIL로 포함한다. `전체 거절`에는 별도 스텝 분류기의 불일치도 포함된다.

| 기종 | 정상 사진 | 분류 정답 | ROI 검출 | U-Net 정답 스텝 PASS | 전체 정상 PASS | 다른 스텝 U-Net 거절 | 다른 스텝 전체 거절 |
|---|---:|---:|---:|---:|---:|---:|---:|
| C003 | 2280 | 2272/2280 | 2278/2280 | 2243/2280 | **2239/2280 (98.20%)** | 25077/25080 (99.99%) | 25077/25080 (99.99%) |

다른 스텝의 전체 거절률은 분류기 불일치 조건을 포함하므로 **U-Net 단독 거절률과 함께** 읽어야 한다. 스텝별 세부 건수와 오수용 조합은 C003 `summary.json`에 있다. U-Net 단독 다른 스텝 오수용 3건은 [원본·ROI 폴더](unet_wrong_step_pass/README.md)에 모았다.

## 2. 정상 FAIL 원인

- `3029C003AA`: 분류 불일치 8장, 분류가 맞았으나 `NO_ROI` 2장, 분류·ROI가 맞았으나 U-Net FAIL 31장.

## 3. NG 시험

기존 `NG_p` 확정 사진 4장과 사용자가 정상 목록에서 재확인한 NG 1장을 **C003 test 정상 분모와 분리**해 평가했다. C009 NG 두 장은 별도로 학습된 C009 모델의 판정이며 C003 모델의 교차 기종 NG 시험이 아니다. NG의 기대 스텝은 파일명/검토 라벨이다.

| 기종 | 시리얼 | NG 출처 | 기대 스텝 | 분류 스텝 | ROI | U-Net 기대 스텝 | 최종 |
|---|---|---|---:|---:|---|---|---|
| 3029C003AA | 2EQ16580 | ng_folder | 9 | 10 | 검출 | FAIL | **FAIL** |
| 3029C003AA | 2EQ17023 | ng_folder | 7 | 1 | NO_ROI | FAIL | **FAIL** |
| 3029C003AA | 2EQ16854 | ng_reclassified | 11 | 12 | 검출 | FAIL | **FAIL** |
| 3029C009AA | 2EX36524 | ng_folder | 3 | 3 | 검출 | PASS | **PASS** |
| 3029C009AA | 2EX36613 | ng_folder | 3 | 3 | 검출 | FAIL | **FAIL** |

NG 표본은 5장뿐이므로 위 FAIL 비율을 일반 결함 검출률로 해석하지 않는다. `NO_ROI`의 FAIL은 U-Net 이상점수에 의한 검출이 아니다.

## 4. 시각화

- [단계별 정상 수용률](figures/stage_rates.png)
- [스텝별 최종 정상 PASS](figures/normal_pass_by_step.png)
- [다른 스텝 U-Net 오수용 조합](figures/wrong_step_unet_heatmap.png)
- [NG별 최종 판정](figures/ng_decisions.png)
- [C003 정상 FAIL 41장 전체 갤러리](normal_final_fail/README.md)
- [C003 정상 FAIL 일부 사례](figures/normal_fail_gallery_3029C003AA.jpg)
- [C003의 다른 스텝 U-Net 오수용 원본·ROI](unet_wrong_step_pass/README.md)
- [다른 스텝 최종 오수용 3장](figures/wrong_step_final_pass_gallery.jpg)
- [C003·C009 NG 원본·ROI](figures/ng_image_gallery.jpg)

이미지별 원본 경로·스텝 예측·ROI 박스·점수는 `3029C003AA/per_image.csv`에 있다. NG 별도 사례는 `ng_examples.csv`에 있다. 보류 test 사진은 이번 평가 전 U-Net 학습이나 임계값 선택에 사용하지 않았다.
