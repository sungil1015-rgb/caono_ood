# ROI U-Net 대조학습 결과

설정상 학습 목표는 20에폭이다. 실제 완료 에폭은 3029C003AA: 17, 3029C009AA: 20. 스텝 1~12에서 스텝당 12장씩 총 144장을 한 SupCon 배치로 사용했다. ResNet18 인코더는 기존 체크포인트에 고정하고 U-Net 디코더와 투영 헤드만 학습했다. NG·검증 이미지는 학습에 쓰지 않았다.

평가는 알려진 스텝의 학습 정상 임베딩 5개까지의 평균 코사인 거리와 스텝별 val_normal p99 임계값을 사용했다. 검출 실패는 NO_ROI로 따로 집계했다.

| 기종 | 모델 | 깊이 | 독립 정상 PASS | NG 점수 FAIL | NG NO_ROI | 대응 정상 PASS |
|---|---|---|---:|---:|---:|---:|
| 3029C003AA | new U-Net | bottleneck | 1082/1104 | 1/2 | 1 | 2/2 |
| 3029C003AA | new U-Net | up2 | 1081/1104 | 1/2 | 1 | 2/2 |
| 3029C003AA | new U-Net | up1 | 1080/1104 | 1/2 | 1 | 2/2 |
| 3029C003AA | previous U-Net | bottleneck | 1092/1104 | 1/2 | 1 | 2/2 |
| 3029C003AA | previous U-Net | up2 | 1091/1104 | 1/2 | 1 | 2/2 |
| 3029C003AA | previous U-Net | up1 | 1087/1104 | 1/2 | 1 | 2/2 |
| 3029C003AA | previous ROI contrastive | stem | 1078/1104 | 1/2 | 1 | 2/2 |
| 3029C003AA | previous ROI contrastive | layer1 | 1085/1104 | 1/2 | 1 | 2/2 |
| 3029C003AA | previous ROI contrastive | layer2 | 1089/1104 | 1/2 | 1 | 2/2 |
| 3029C003AA | previous ROI contrastive | layer3 | 1092/1104 | 1/2 | 1 | 2/2 |
| 3029C009AA | new U-Net | bottleneck | 317/324 | 1/2 | 0 | 1/2 |
| 3029C009AA | new U-Net | up2 | 316/324 | 1/2 | 0 | 2/2 |
| 3029C009AA | new U-Net | up1 | 307/324 | 1/2 | 0 | 2/2 |
| 3029C009AA | previous U-Net | bottleneck | 316/324 | 1/2 | 0 | 2/2 |
| 3029C009AA | previous U-Net | up2 | 310/324 | 1/2 | 0 | 1/2 |
| 3029C009AA | previous U-Net | up1 | 310/324 | 1/2 | 0 | 0/2 |
| 3029C009AA | previous ROI contrastive | stem | 310/324 | 0/2 | 0 | 1/2 |
| 3029C009AA | previous ROI contrastive | layer1 | 307/324 | 0/2 | 0 | 0/2 |
| 3029C009AA | previous ROI contrastive | layer2 | 299/324 | 1/2 | 0 | 2/2 |
| 3029C009AA | previous ROI contrastive | layer3 | 304/324 | 1/2 | 0 | 2/2 |

새 학습과 이전 학습은 양성 쌍 수가 달라 손실값끼리 직접 비교하지 않는다. 검증 정상 수용률을 먼저 비교한다. 테스트 분할은 사용하지 않았다.

하이퍼파라미터: `hyperparameters.toml`; 이미지별 점수: 각 기종의 `results.json`; 스텝별 정상 실패 수: `summary.json`.
