#!/usr/bin/env bash
# Downloads the accuracy-test data into tests/accuracy/data (or $ACC_DATA).
set -euo pipefail
D=${ACC_DATA:-$(dirname "$0")/data}
mkdir -p "$D" && cd "$D"
[ -d coco128 ] || { curl -fsSL -o coco128.zip https://github.com/ultralytics/assets/releases/download/v0.0.0/coco128.zip && unzip -q coco128.zip && rm coco128.zip; }
[ -f vtest.avi ] || curl -fsSL -o vtest.avi https://raw.githubusercontent.com/opencv/opencv/4.x/samples/data/vtest.avi
ls -la
