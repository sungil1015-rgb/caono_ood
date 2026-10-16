# 고정 인코더 + U-Net 디코더 ROI 대조학습

기존 ROI 대조학습의 기종별 ResNet18 인코더 체크포인트를 **고정**했다. `layer3→layer2→layer1`을 U-Net 디코더로 결합하고 새 투영 헤드와 디코더만 정상 ROI로 8에폭 학습했다. 입력 1024×1024, 스텝 1~12, 스텝당 6장·총 72장 SupCon 배치, 90도 단위 회전과 약한 색 증강을 유지했다. ROI의 회색 패딩은 특징 풀링에서 제외했다.

72장 전체의 SupCon 손실을 계산하면서 GPU 메모리를 줄이기 위해 디코더를 작은 단위로 두 번 통과시켰다. 첫 통과에서 전체 임베딩에 대한 손실 기울기를 구하고 두 번째 통과에서 디코더를 업데이트했다. NG 및 검증 이미지는 학습에 사용하지 않았다.

## 독립 정상과 확정 NG (스텝별 `val_normal` p99 임계값)

| 기종 | 모델·깊이 | 독립 정상 PASS | NG 점수 FAIL | NG `NO_ROI` | 대응 정상 PASS |
|---|---|---:|---:|---:|---:|
| 3029C003AA | U-Net bottleneck | 1092/1104 | 1/2 | 1 | 2/2 |
| 3029C003AA | U-Net up2 | 1091/1104 | 1/2 | 1 | 2/2 |
| 3029C003AA | U-Net up1 | 1087/1104 | 1/2 | 1 | 2/2 |
| 3029C003AA | 기존 stem | 1078/1104 | 1/2 | 1 | 2/2 |
| 3029C003AA | 기존 layer1 | 1085/1104 | 1/2 | 1 | 2/2 |
| 3029C003AA | 기존 layer2 | 1089/1104 | 1/2 | 1 | 2/2 |
| 3029C003AA | 기존 layer3 | 1092/1104 | 1/2 | 1 | 2/2 |
| 3029C009AA | U-Net bottleneck | 316/324 | 1/2 | 0 | 2/2 |
| 3029C009AA | U-Net up2 | 310/324 | 1/2 | 0 | 1/2 |
| 3029C009AA | U-Net up1 | 310/324 | 1/2 | 0 | 0/2 |
| 3029C009AA | 기존 stem | 310/324 | 0/2 | 0 | 1/2 |
| 3029C009AA | 기존 layer1 | 307/324 | 0/2 | 0 | 0/2 |
| 3029C009AA | 기존 layer2 | 299/324 | 1/2 | 0 | 2/2 |
| 3029C009AA | 기존 layer3 | 304/324 | 1/2 | 0 | 2/2 |

U-Net의 더 관대한 `val_normal` 최댓값 임계값 기준 정상 PASS:

- 3029C003AA bottleneck: 1095/1104
- 3029C003AA up2: 1098/1104
- 3029C003AA up1: 1098/1104
- 3029C009AA bottleneck: 317/324
- 3029C009AA up2: 313/324
- 3029C009AA up1: 311/324

## C009 확정 NG 점수

| 시리얼 | U-Net bottleneck 점수/임계값 | 판정 |
|---|---:|---|
| 2EX36524 | 0.583 | PASS |
| 2EX36613 | 4.842 | FAIL |

C003에서 U-Net `bottleneck`의 정상 PASS는 기존 `layer3`와 같은 1092/1104다. C009는 U-Net `bottleneck` 316/324로 기존의 최고 정상 PASS 310/324보다 6장 많다. 그러나 두 기종 모두 독립 정상 전부 PASS에는 이르지 못했다.

더 높은 해상도의 `up1`은 C009 정상 PASS 310/324, 대응 정상 0/2로 오히려 나빠졌다. 작은 문자 누락 NG `2EX36524`의 `bottleneck` 점수/임계값은 0.583, 같은 시리얼 정상 사진은 0.814이다. 이 출력에서는 누락 사진의 점수가 정상보다 낮아 임계값만으로 둘을 구분할 수 없다. 따라서 U-Net 디코더 추가만으로 작은 누락을 검출했다는 근거는 없다.

이 비교는 기존 ROI 대조학습 인코더를 다시 사용해 디코더를 추가 학습한 결과다. 추가 학습량과 패딩 제외 풀링도 달라졌으므로 수치 차이를 U-Net 구조만의 효과로 해석할 수 없다.


이미지별 점수는 각 기종의 `results.json`, 스텝별 정상 실패 수는 `summary.json`에 있다. C003 `val_step`에서 검출기 학습 이력이 있는 시리얼 3대(36장)는 기존 ROI 매니페스트와 동일하게 제외했다. 테스트 분할은 사용하지 않았다.

## 재현

```bash
canon/.venv/bin/python canon/experiment/train_roi_unet_contrastive.py --product 3029C003AA --device cuda:1
canon/.venv/bin/python canon/experiment/train_roi_unet_contrastive.py --product 3029C009AA --device cuda:0
canon/.venv/bin/python canon/experiment/evaluate_roi_unet_contrastive.py --product 3029C003AA --device cuda:1
canon/.venv/bin/python canon/experiment/evaluate_roi_unet_contrastive.py --product 3029C009AA --device cuda:0
canon/.venv/bin/python canon/experiment/summarize_roi_unet_contrastive.py
```
