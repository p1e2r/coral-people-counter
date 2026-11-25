# Accuracy tests

Run against real data with the app's real detector, tracker and counter (CPU, same weights as the
Coral models). CI runs them inside the published image after every build (`accuracy` workflow).

```bash
bash tests/accuracy/fetch_data.sh
MODEL_DIRS=/path/to/models python tests/accuracy/count_eval.py   # counting on real video
MODEL_DIRS=/path/to/models python tests/accuracy/det_eval.py     # person detection on COCO128
```

## Counting: real pedestrian video

OpenCV's `vtest.avi`: an angled camera over a plaza, 768×576, 10 fps, 795 frames.
**31 people** cross a vertical line at x = 45 %. The ground truth was counted by hand from a
slit-scan of the line, and the unclear cases were checked frame by frame.

| Model (threshold 0.45) | Counted | Correct | Extra | Missed | F1 |
|---|---|---|---|---|---|
| SSD MobileNet v2 | 23 | 23 | 0 | 8 | 85.2 % |
| SSDLite MobileDet | 28 | 28 | 0 | 3 | 94.9 % |
| EfficientDet-Lite0 | 26 | 26 | 0 | 5 | 91.2 % |
| **EfficientDet-Lite1 (default)** | **31** | **31** | **0** | **0** | **100 %** |
| EfficientDet-Lite2 | 31 | 31 | 0 | 0 | 100 % |
| EfficientDet-Lite3x | 27 | 26 | 1 | 5 | 89.7 % |

## Detection: COCO128 (254 people in 128 photos)

Each model is shown at its own best threshold. These figures are for people at least 60 px tall.

| Model | AP50 (all) | Best F1 | Threshold | Precision | Recall |
|---|---|---|---|---|---|
| SSD MobileNet v2 | 43.1 | 78.6 | 0.25 | 87.1 | 71.6 |
| SSDLite MobileDet | 51.7 | 81.9 | 0.50 | 85.4 | 78.7 |
| EfficientDet-Lite1 | 60.1 | 84.2 | 0.25 | 87.1 | 81.6 |
| EfficientDet-Lite2 | 62.6 | 84.8 | 0.35 | 97.2 | 75.2 |
| EfficientDet-Lite3x | 67.4 | 87.3 | 0.20 | 92.1 | 83.0 |

## Limits

- The counting test has 31 crossings from one camera at one time of day. A 1–2 crossing difference
  between models is within noise. Check your own camera with the **Recent counts** snapshots.
- The top-down model has no labelled test set here, so it isn't covered by these tests.
- Coral results can differ very slightly from CPU because the Edge TPU compiler is involved,
  but the weights are the same quantized ones.
