# 주요 실험 결과

| 폴더 | 내용 |
|---|---|
| [`initial_unet/`](initial_unet/) | 72장 배치의 초기 U-Net과 기존 ROI 인코더 비교 |
| [`batch144/`](batch144/) | 144장 배치 U-Net의 C003·C009 결과 |
| [`color_ablation/`](color_ablation/) | 회전 제외·색 증강·손실 가중치 비교 |
| [`rotation_ablation/`](rotation_ablation/) | 동일 조건에서 회전 증강 유무 비교와 부트스트랩 구간 |
| [`selected_c003/`](selected_c003/) | 현재 선택한 C003 U-Net의 전체 이미지별 점수, 학습 손실, 실행 설정 |
| [`c009_batch144/`](c009_batch144/) | C009 144장 배치의 전체 이미지별 점수 |
| [`cross_step/`](cross_step/) | C003 상위 두 모델의 12개 스텝 전체 입력 시험과 이미지별 CSV |
| [`overview/`](overview/) | 전체 Canon 실험 맥락을 포함한 상세 보고서 |
| [`test_pipeline/`](test_pipeline/) | C003 보류 test 2,280장의 분류→ROI→U-Net 평가 및 C003·C009 NG 별도 사례와 그래프·갤러리 |

교차 스텝 시험의 `summary.json`과 `per_image.csv`는 **시험 당시 원래 라벨**을 그대로 보존한다. `2EQ16854/step_11.jpg`가 실제 NG라는 사용자 재확인을 반영한 해석은 [`cross_step/summary.md`](cross_step/summary.md)에 적었다. 최신 U-Net의 별도 보류 test 분할 결과는 [`test_pipeline/summary.md`](test_pipeline/summary.md)에 있다.
