# C003 정상 폴더 이미지의 최종 FAIL 41장

보류 test 정상 폴더에서 최종 FAIL된 41장을 전부 모았다. 스텝 분류 불일치 8장, ROI 미검출 2장, 분류와 ROI가 성공했지만 U-Net이 거절한 이미지 31장이다. 이 분류는 최종 FAIL의 첫 원인을 기준으로 한다.

갤러리 왼쪽은 원본 사진이며 빨간 사각형은 검출된 ROI 박스다. 오른쪽은 모델에 넣은 ROI다. 파일명과 `manifest.csv`에는 실제 스텝, 예측 스텝, U-Net 점수와 임계값이 담겨 있다. `NO_ROI`는 ROI와 U-Net 점수가 없다. 정상 폴더의 라벨은 전체 이미지를 사람이 다시 확인한 결과는 아니다.

## 원인별 갤러리

- [STEP_MISMATCH 1–8](./step_mismatch_gallery_01.jpg)
- [NO_ROI 1–2](./no_roi_gallery_01.jpg)
- [UNET_FAIL 1–8](./unet_fail_gallery_01.jpg)
- [UNET_FAIL 9–16](./unet_fail_gallery_02.jpg)
- [UNET_FAIL 17–24](./unet_fail_gallery_03.jpg)
- [UNET_FAIL 25–31](./unet_fail_gallery_04.jpg)
