# C003 회전 제외·강한 색 증강 실험

두 새 실험은 스텝당 12장·총 144장으로 20에폭을 목표로 했으나, 손실 정체와 사용자 요청에 따라 7에폭에서 멈췄다. 공통으로 회전을 제거하고 색 증강 확률 0.85, 밝기·대비 0.15, 채도 0.08을 썼다. 손실 가중치만 1:1:1과 3:1:1로 달리했다. 
이전 모델은 회전 증강과 약한 색 증강을 사용했고 17에폭에서 중단했다. 따라서 이전 모델과의 차이는 증강·학습 길이가 함께 반영된다.

임계값은 각 모델에서 스텝별 `val_normal` p99로 보정했다. 독립 정상 `val_step` 수용률을 우선 비교한다.

| 설정 | 완료 에폭 | 깊이 | 독립 정상 PASS | NG 점수 FAIL | NG NO_ROI | 대응 정상 PASS |
|---|---:|---|---:|---:|---:|---:|
| previous rotation + mild color | 17 | bottleneck | 1082/1104 | 1/2 | 1 | 2/2 |
| previous rotation + mild color | 17 | up2 | 1081/1104 | 1/2 | 1 | 2/2 |
| previous rotation + mild color | 17 | up1 | 1080/1104 | 1/2 | 1 | 2/2 |
| color only, equal loss | 7 | bottleneck | 1094/1104 | 1/2 | 1 | 2/2 |
| color only, equal loss | 7 | up2 | 1093/1104 | 1/2 | 1 | 2/2 |
| color only, equal loss | 7 | up1 | 1097/1104 | 1/2 | 1 | 2/2 |
| color only, bottleneck 3:1:1 | 7 | bottleneck | 1093/1104 | 1/2 | 1 | 2/2 |
| color only, bottleneck 3:1:1 | 7 | up2 | 1087/1104 | 1/2 | 1 | 2/2 |
| color only, bottleneck 3:1:1 | 7 | up1 | 1086/1104 | 1/2 | 1 | 2/2 |

같은 7에폭에서 색 증강만 적용한 모델의 up1은 1097/1104장, bottleneck 가중치를 3배로 높인 모델의 up1은 1086/1104장을 PASS시켰다. 추가 가중치의 이점은 관찰되지 않았다.

NG 및 대응 정상의 이미지별 점수는 각 설정의 `results.json`에 있다. NG는 임계값 설정이나 모델 선택에 사용하지 않았다. 보류된 test 분할은 사용하지 않았다.
