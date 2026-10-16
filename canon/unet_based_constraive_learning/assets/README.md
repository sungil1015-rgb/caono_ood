# 실행 보조 자산

| 항목 | 용도 |
|---|---|
| `splits/` | 학습·검증·보류 test 이미지 경로와 라벨 CSV |
| `roi_manifest.json` | 기종·스텝별 ROI scale과 검출된 ROI 메타데이터 |
| `crop_canon_data/` | ROI 검출기 학습 시리얼 중복을 검사할 YOLO 라벨 |
| `fonts/` | 결과 갤러리용 Nanum Gothic 글꼴 |
| `ng_reference/3029C009AA.json` | 별도 C009 평가에서 사용한 NG 두 장의 판정 정보 |
| `ng_reference/3029C009AA/` | 위 NG 두 장의 원본과 ROI 이미지 |

원본 Canon 이미지 데이터는 용량 때문에 포함하지 않는다. 단일 이미지 판정은 실행할 이미지 경로를 직접 입력하며, 보류 test 전체 평가와 재학습은 `CANON_DATA_ROOT`로 데이터 루트를 지정한다.
