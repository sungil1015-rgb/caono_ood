# 실행 코드

이 디렉터리의 Python 모듈끼리 import하며, 설정·가중치·메타데이터는 상위 디렉터리에서 읽는다. 원본 이미지가 필요한 일괄 평가와 재학습은 `CANON_DATA_ROOT`로 데이터 루트를 지정한다. 경로 지정은 [bundle_paths.py](bundle_paths.py)에 모았다.

| 파일 | 역할 |
|---|---|
| [predict_image.py](predict_image.py) | 기종·기대 스텝을 입력받아 한 장을 분류→ROI→U-Net으로 판정 |
| [evaluate_unet_test_pipeline.py](evaluate_unet_test_pipeline.py) | 보류 test 전체와 NG 평가 |
| [image_preprocess.py](image_preprocess.py), [known_product_step.py](known_product_step.py) | 회전 전처리와 기종별 스텝 분류기 |
| [train_step_roi_detector.py](train_step_roi_detector.py), [prepare_roi_contrastive.py](prepare_roi_contrastive.py) | 공용 ROI 검출기 구조와 ROI 매니페스트 생성 |
| [train_step_contrastive.py](train_step_contrastive.py), [train_roi_step_contrastive.py](train_roi_step_contrastive.py) | 고정 인코더의 원래 대조학습 코드 |
| [train_roi_unet_contrastive.py](train_roi_unet_contrastive.py), [evaluate_roi_unet_contrastive.py](evaluate_roi_unet_contrastive.py) | U-Net 디코더 학습과 점수 평가 |
| [train_roi_resnet18_full_supcon.py](train_roi_resnet18_full_supcon.py), [evaluate_roi_resnet18_full_supcon.py](evaluate_roi_resnet18_full_supcon.py) | 비교 ResNet18 모델 |
| [evaluate_roi_cross_step_top2.py](evaluate_roi_cross_step_top2.py) | U-Net과 비교 모델의 교차 스텝 평가 |
| [summarize_unet_test_pipeline.py](summarize_unet_test_pipeline.py), [render_unet_test_gallery.py](render_unet_test_gallery.py) | 집계 그래프와 원본·ROI 갤러리 |
| [collect_unet_wrong_step_pass.py](collect_unet_wrong_step_pass.py), [collect_unet_normal_final_fail.py](collect_unet_normal_final_fail.py) | 오수용·정상 FAIL 사진 모음 |

입력 이미지 없이 추론 구조와 가중치를 확인하려면 상위 [README.md](../README.md)의 한 장 판정 명령을 참고한다. 전체 평가에는 원본 `data/canon_data`가 필요하다. 학습용 ROI를 재생성했다면 `CANON_ROI_MANIFEST`로 새 매니페스트 경로를 지정한다.
