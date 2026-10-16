# ROI 기반 U-Net 대조학습

기종과 기대 스텝을 알고 있을 때 **스텝 분류 → ROI 검출·crop → U-Net 점수**로 PASS/FAIL을 판단하는 실행 묶음이다. C003 선택 모델은 고정 ResNet18 인코더와 U-Net 디코더의 `up1` 출력을 사용한다. C009는 별도로 학습된 모델이다.

| 폴더 | 내용 |
|---|---|
| [model_code](model_code/) | 독립 실행 가능한 학습·추론·평가 코드 |
| [models](models/) | 스텝 분류기, ROI 검출기, 인코더, 디코더, 정상 특징 저장소와 임계값 |
| [configs](configs/) | 체크포인트와 일치하는 TOML 설정 |
| [assets](assets/) | 분할 CSV, ROI 크기 정보, 검출기 라벨 메타데이터, 글꼴, C009 NG 사례 |
| [hyperparameters](hyperparameters/) | 실험별 설정 보관본 |
| [model_structure](model_structure/) | 모델 구조 설명 |
| [experiment_results](experiment_results/) | 당시 평가 결과와 시각화 |
| `outputs/` | 새 추론과 평가 결과가 저장되는 위치. Git에서 제외된다. |

## 설치

Python 3.12.3 환경에서 확인한 패키지는 [requirements.txt](requirements.txt)에 적었다. 새 저장소에 가중치를 올릴 때는 Git LFS가 필요하다.

```bash
conda create -n canon-unet python=3.12
conda activate canon-unet
python -m pip install -r requirements.txt
git lfs pull
```

CUDA를 쓰려면 설치한 PyTorch가 해당 GPU를 지원해야 한다. CPU에서는 아래 명령의 `--device cpu`를 사용한다.

## 이미지 한 장 판정

새 저장소의 루트에서 실행한다. 이미지 파일만 있으면 되고 원래 Canon 데이터 폴더는 필요하지 않다.

```bash
python model_code/predict_image.py /path/to/image.jpg \
  --product 3029C003AA --expected-step 11 --device cpu
```

`3029C009AA`도 같은 명령에서 사용할 수 있다. JSON 결과와 검출된 ROI는 `outputs/predict_image/`에 저장된다. 최종 PASS는 분류 스텝이 기대 스텝과 같고, ROI가 검출되며, U-Net 점수가 해당 스텝 임계값 이내일 때만 나온다.

## 보류 test 전체 재실행

전체 평가는 원본 이미지 데이터가 추가로 필요하다. `CANON_DATA_ROOT`에는 `data/canon_data` 폴더를 가진 디렉터리를 지정한다. 분할 CSV와 모델 가중치는 이 저장소에 포함되어 있다.

```bash
export CANON_DATA_ROOT=/path/to/canon
python model_code/evaluate_unet_test_pipeline.py \
  --product 3029C003AA --device cuda:0 --workers 4
python model_code/summarize_unet_test_pipeline.py
python model_code/render_unet_test_gallery.py
python model_code/collect_unet_normal_final_fail.py
python model_code/collect_unet_wrong_step_pass.py
```

위 보고서의 정상 수용률과 다른 스텝 거절률은 C003 보류 test만 집계한다. C009 NG 두 장은 [번들 자산](assets/ng_reference/)의 기존 판정으로 별도 표시한다. ROI를 새 데이터에서 다시 추출해 학습할 때는 `model_code/prepare_roi_contrastive.py`로 매니페스트를 만들고, 생성된 파일을 `CANON_ROI_MANIFEST` 환경 변수에 지정한다.

## 기록된 결과

선택 C003 모델은 ROI가 검출된 정상 학습 이미지 6,829장으로 학습했다. 기존 `val_step` 정상 라벨 1,104장 중 1,097장 PASS였다. 이후 확인된 NG 한 장을 정상 집계에서 제외하면 1,097/1,103장 PASS다. 다른 스텝 입력 결과는 [교차 스텝 보고서](experiment_results/cross_step/summary.md)에 있다.

C003 보류 test의 최종 정상 PASS는 **2,239/2,280장**, 다른 스텝 최종 거절은 **25,077/25,080건**이다. C009 정상 test는 현행 지표에서 제외했고, C009 NG 두 장은 C009 전용 모델의 별도 사례로 유지했다. 자세한 수치는 [test 파이프라인 보고서](experiment_results/test_pipeline/summary.md)에 있다.
