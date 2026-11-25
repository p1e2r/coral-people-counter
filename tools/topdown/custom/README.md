# Use your own top-down model (optional)

By default the image build downloads the Zenodo top-view YOLOv8n model.
To use your own instead:

- `model.pt` here: an Ultralytics YOLO model (v5u/v8/11) where class 0 = person.
  Fine-tuning on frames from *your* camera gives the best accuracy.
- `calib/`: 100–300 frames from your camera (jpg/png, no labels needed) for INT8 calibration.

Then rebuild: `docker compose build`.

You can also skip rebuilding: put `topdown_edgetpu.tflite` (and optionally `topdown.tflite`)
into `./data/models/` on the host. Files there override the built-in ones.
