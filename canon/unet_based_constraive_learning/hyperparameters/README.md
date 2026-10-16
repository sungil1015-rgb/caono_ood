# TOML 설정 안내

| 파일 | 실험 |
|---|---|
| `roi_unet_contrastive_v1.toml` | 첫 72장 배치 U-Net 실험 |
| `roi_unet_contrastive_batch144_epoch20.toml` | 스텝당 12장, 144장 배치 실험 |
| `roi_unet_c003_color_only_equal.toml` | **선택 모델**: 회전 없는 색 증강, 세 출력 손실 1:1:1 |
| `roi_unet_c003_color_only_bottleneck3.toml` | bottleneck 손실 비중 3:1:1 비교 |
| `roi_unet_c003_strong_color_rotation_control.toml` | 색 증강은 동일하고 회전 증강을 넣은 대조군 |
| `roi_unet_rotation_compare.toml` | 회전 대조 결과 분석 설정 |
| `roi_unet_fusion_c003.toml` | 여러 출력의 점수 결합 탐색 설정 |
| `selected_run_hyperparameters.toml` | 선택 체크포인트 폴더에 보관된 학습 당시 TOML 사본 |

선택 실험의 TOML은 최대 20에폭을 설정하지만 실제 체크포인트는 **7에폭**까지 학습됐다. 실제 완료 에폭은 [`../experiment_results/selected_c003/history.json`](../experiment_results/selected_c003/history.json)을 참조한다. 이 복사본의 `output_dir`은 원본 `canon/` 프로젝트 기준 상대 경로다.
