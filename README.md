# ROI 기반 U-Net 대조학습 정리

이 폴더는 `3029C003AA`와 `3029C009AA`의 스텝 1~12 ROI 대조학습 실험에서 사용한 설정, 모델 구현, 구조 설명, 주요 결과를 주제별로 모은 자료다. **현재 선택한 C003 모델은 `color_only_equal` 설정의 U-Net `up1` 출력**이다.

| 폴더 | 내용 |
|---|---|
| [`hyperparameters/`](hyperparameters/) | 실제 학습·비교에 사용한 TOML 설정. `selected_run_hyperparameters.toml`은 선택 모델 체크포인트에 저장된 설정의 복사본이다. |
| [`model_structure/`](model_structure/) | 인코더, U-Net 디코더, 세 출력, 손실과 판정 절차 설명. |
| [`model_code/`](model_code/) | 학습·평가 코드와 필요한 로컬 의존 코드의 복사본. |
| [`experiment_results/`](experiment_results/) | C003·C009 평가, 증강·손실 비교, 다른 스텝 입력 시험 및 이미지별 결과. |

선택 모델은 C003에서 ROI가 검출된 정상 학습 사진 **6,829장**으로 학습했다. `val_step`의 기존 정상 라벨 1,104장 중 **1,097장 PASS**, 다른 스텝 입력 12,144건 중 **12,143건 거절**이었다. 사용자가 다른 스텝 오수용에 포함된 `2EQ16854/step_11.jpg`를 NG라고 확인했다. 이를 정상 집계에서 빼면 확인된 정상 **1,103장 중 1,097장 PASS**, 그 정상 사진의 다른 스텝 입력 **12,133건 전부 거절**이다. 이 NG는 예상 스텝 11에서 FAIL, 스텝 12 기준에서는 PASS이므로 운영 시 예상 스텝을 사용해야 한다. 자세한 수치는 [`experiment_results/cross_step/summary.md`](experiment_results/cross_step/summary.md)에 있다.

## Conda 실행 환경

실험에 사용한 Python 3.12.3 기준의 외부 패키지는 [`requirements.txt`](requirements.txt)에 정리했다.

```bash
conda create -n canon-unet python=3.12
conda activate canon-unet
python -m pip install -r canon/unet_based_constraive_learning/requirements.txt
```

CUDA를 쓸 경우 설치 환경에 맞는 PyTorch 실행 장치 지원을 확인한다. CPU 환경에서도 동일한 requirements를 사용할 수 있다.

## 원본 코드와 가중치

여기의 Python 파일은 검토하기 편하도록 복사한 **코드 스냅샷**이다. 코드 안의 상대 경로와 `ROOT` 계산은 원래 `canon/experiment/` 배치를 기준으로 작성되었으므로, 실행할 때는 아래 원본 경로를 사용한다. U-Net은 이미 학습된 ROI 대조학습 ResNet18 인코더를 고정해 사용한다.

| 항목 | 원본 위치 |
|---|---|
| U-Net 학습·평가 코드 | `../experiment/train_roi_unet_contrastive.py`, `../experiment/evaluate_roi_unet_contrastive.py` |
| 선택 모델 TOML | `../experiment/configs/roi_unet_c003_color_only_equal.toml` |
| 고정 인코더 가중치 | `../data/experiment_v1/roi_contrastive/3029C003AA/encoder_weights.pt` |
| 선택 U-Net 가중치 | `../data/experiment_v1/roi_unet_contrastive/color_only_equal_batch144_epoch20/3029C003AA/decoder_weights.pt` |
| 정상 특징 저장소 | `../data/experiment_v1/roi_unet_contrastive/color_only_equal_batch144_epoch20/3029C003AA/train_normal_bank.pt` |

```bash
canon/.venv/bin/python canon/experiment/evaluate_roi_unet_contrastive.py \
  --product 3029C003AA \
  --config canon/experiment/configs/roi_unet_c003_color_only_equal.toml \
  --device cuda:0
canon/.venv/bin/python canon/experiment/evaluate_roi_cross_step_top2.py --device cuda:0
```

TOML의 `epochs = 20`은 **계획한 최대 에폭**이고, 선택 체크포인트는 손실 정체로 **7에폭**에서 종료한 결과다. 완료 에폭과 손실은 [`experiment_results/selected_c003/history.json`](experiment_results/selected_c003/history.json)에 기록돼 있다. 이후 C003 보류 test 2,280장에 스텝 분류기→ROI→U-Net을 적용했다. 최종 정상 PASS는 **2,239/2,280장**, 다른 스텝 최종 거절은 **25,077/25,080건**이다. C009 정상 test는 현재 지표에서 제외했으며, C009 NG 두 장은 C009 전용 모델의 별도 사례로 유지했다. 결과와 그림은 [`experiment_results/test_pipeline/summary.md`](experiment_results/test_pipeline/summary.md)에 있다.
