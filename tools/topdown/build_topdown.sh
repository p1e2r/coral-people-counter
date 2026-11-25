#!/usr/bin/env bash
# Builds the top-down (overhead camera) person model for the Coral Edge TPU.
#
# Source model: "YOLO-based models trained via transfer learning with top-view person
# image annotations" — Hernando Cánovas, Martínez-Sala, Aarnoutse-Sánchez, Alcaráz-Espín,
# Universidad Politécnica de Cartagena (2025). DOI 10.5281/zenodo.15731095. CC BY 4.0.
#
# Output (in $OUT): topdown_edgetpu.tflite, topdown.tflite (CPU fallback), TOPDOWN_MODEL.txt
# Never fails the image build: on any error it leaves $OUT/TOPDOWN_FAILED.txt instead.
set -uo pipefail
OUT=${OUT:-/out}
MODEL=${TOPDOWN_MODEL:-Yolo_v8_n}
IMGSZ=${TOPDOWN_IMGSZ:-320}
BASE=${TOPDOWN_URL:-https://zenodo.org/records/15731095/files}
mkdir -p "$OUT" /work && cd /work

fail() { echo "top-down model build failed: $*" | tee "$OUT/TOPDOWN_FAILED.txt"; exit 0; }
command -v yolo >/dev/null || fail "export tools (ultralytics/tensorflow) failed to install"

if [ -f /src/model.pt ]; then
  cp /src/model.pt model.pt            # your own trained weights, if provided
else
  curl -fsSL --retry 3 -o model.pt "$BASE/${MODEL}.pt?download=1" || fail "could not download ${MODEL}.pt"
fi

# INT8 calibration images: the record's top-view test images (labels not needed)
mkdir -p calib/images
if curl -fsSL --retry 3 -o images.zip "$BASE/images.zip?download=1" && unzip -q -o images.zip -d raw; then
  find raw -type f \( -iname '*.jpg' -o -iname '*.jpeg' -o -iname '*.png' \) -exec cp {} calib/images/ \;
fi
[ -d /src/calib ] && find /src/calib -type f \( -iname '*.jpg' -o -iname '*.png' \) -exec cp {} calib/images/ \;
n=$(ls calib/images | wc -l)
if [ "$n" -gt 0 ]; then
  printf "path: /work/calib\ntrain: images\nval: images\nnames:\n  0: person\n" > calib.yaml
  DATA="data=calib.yaml"
  echo "calibrating with $n top-view images"
else
  DATA=""   # falls back to Ultralytics' default calibration set
fi

yolo export model=model.pt format=edgetpu imgsz="$IMGSZ" $DATA || fail "export/compile failed"
tpu=$(ls model_saved_model/*_full_integer_quant_edgetpu.tflite 2>/dev/null | head -1)
cpu=$(ls model_saved_model/*_full_integer_quant.tflite 2>/dev/null | head -1)
[ -n "$tpu" ] && [ -n "$cpu" ] || fail "compiled model not found"
cp "$tpu" "$OUT/topdown_edgetpu.tflite"
cp "$cpu" "$OUT/topdown.tflite"
cp model_saved_model/*_edgetpu.log "$OUT/topdown_edgetpu.log" 2>/dev/null || true
cat > "$OUT/TOPDOWN_MODEL.txt" <<EOF
Top-down person model: ${MODEL} (${IMGSZ}x${IMGSZ}), compiled for Coral Edge TPU.
Source: Hernando Cánovas L., Martínez-Sala A. S., Aarnoutse-Sánchez J. C., Alcaráz-Espín J. J.
"YOLO-based models trained via transfer learning with top-view person image annotations",
Universidad Politécnica de Cartagena, 2025. https://doi.org/10.5281/zenodo.15731095
License: Creative Commons Attribution 4.0 International (CC BY 4.0).
Modifications: converted to full-integer TFLite and compiled for the Edge TPU.
EOF
echo "top-down model ready"
