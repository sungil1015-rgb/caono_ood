# 코드 파일 안내

이 폴더의 `.py` 파일은 `canon/experiment/`에 있는 실제 실행 코드를 복사한 스냅샷이다. 경로 계산과 import는 원본 배치에 맞춰져 있으므로, 재실행할 때는 상위 [`README.md`](../README.md)의 원본 경로 명령을 사용한다.

| 파일 | 역할 |
|---|---|
| `prepare_roi_contrastive.py` | 검출기 박스에서 ROI와 마스크용 메타데이터를 만들어 매니페스트 저장 |
| `train_roi_step_contrastive.py` | 고정 인코더로 사용되는 이전 ROI ResNet18 학습과 균형 배치 샘플러 |
| `train_step_contrastive.py` | ResNet18 특징 추출기와 SupCon 손실 함수 정의 |
| `train_roi_unet_contrastive.py` | U-Net 디코더, 패딩 마스크, 전체 배치 손실, 학습 루프 |
| `evaluate_roi_unet_contrastive.py` | 정상 저장소 생성, 스텝별 p99 임계값, 정상·NG 평가 |
| `evaluate_roi_cross_step_top2.py` | 선택 U-Net과 비교 ResNet18의 모든 스텝 입력 시험 |
| `train_roi_resnet18_full_supcon.py`, `evaluate_roi_resnet18_full_supcon.py` | 교차 시험에서 비교한 두 번째 모델의 구현·평가 |
| `evaluate_unet_test_pipeline.py` | 보류 test의 분류→ROI→U-Net 추론, 다른 스텝 입력·NG 평가 |
| `summarize_unet_test_pipeline.py`, `render_unet_test_gallery.py` | C003 정상 지표·그래프와 C003·C009 NG 별도 갤러리 생성 |
| `collect_unet_wrong_step_pass.py` | C003 다른 스텝 U-Net 오수용 원본·ROI 모음 |
| `collect_unet_normal_final_fail.py` | C003 최종 정상 FAIL 전수 원본·ROI와 원인별 갤러리 |

핵심 모델 클래스는 `train_roi_unet_contrastive.py`의 `UNetContrastiveDecoder`, `ConvBlock`, `MaskedProjection`이다.
