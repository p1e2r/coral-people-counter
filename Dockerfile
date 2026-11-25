# Static ffmpeg/ffprobe 7.x (with TLS) for RTSP, RTSPS and HTTP camera streams.
# Multi-arch image, so this works on amd64 and arm64.
FROM mwader/static-ffmpeg:7.1 AS ffmpeg

# Top-down (overhead camera) person model: Zenodo top-view YOLOv8n -> Edge TPU.
# Coral's compiler only runs on x86-64, so this stage is pinned to amd64 even when
# building for arm64; only the small .tflite files are copied into the final image.
# Skip with --build-arg TOPDOWN=0. Never fails the build (see build_topdown.sh).
FROM --platform=linux/amd64 python:3.13-slim AS topdown
ARG TOPDOWN=1
ARG TOPDOWN_MODEL=Yolo_v8_n
ARG TOPDOWN_IMGSZ=320
RUN apt-get update && apt-get install -y --no-install-recommends curl unzip libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*
COPY tools/topdown/requirements.txt /tmp/topdown-requirements.txt
RUN if [ "$TOPDOWN" = "1" ]; then \
      ( pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu \
        && pip install --no-cache-dir -r /tmp/topdown-requirements.txt ) || echo "pip install failed"; \
    fi
COPY tools/topdown/build_topdown.sh /build_topdown.sh
COPY tools/topdown/custom/ /src/
RUN if [ "$TOPDOWN" = "1" ]; then \
      OUT=/out TOPDOWN_MODEL=$TOPDOWN_MODEL TOPDOWN_IMGSZ=$TOPDOWN_IMGSZ bash /build_topdown.sh; \
    else mkdir -p /out && echo "disabled at build time (TOPDOWN=0)" > /out/TOPDOWN_FAILED.txt; fi

# Coral People Counter
# Debian bullseye is used because Google's Coral apt repo (libedgetpu + tflite-runtime)
# ships packages built for its Python 3.9. Works on amd64 and arm64 (Raspberry Pi 4/5).
FROM debian:bullseye-slim

ARG EDGETPU_RUNTIME=libedgetpu1-std
# use libedgetpu1-max for full clock speed (runs hotter)

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    TZ=UTC \
    DATA_DIR=/data \
    PORT=8080

RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates curl gnupg tzdata \
    && curl -fsSL https://packages.cloud.google.com/apt/doc/apt-key.gpg \
        | gpg --dearmor -o /usr/share/keyrings/coral-edgetpu.gpg \
    && echo "deb [signed-by=/usr/share/keyrings/coral-edgetpu.gpg] https://packages.cloud.google.com/apt coral-edgetpu-stable main" \
        > /etc/apt/sources.list.d/coral-edgetpu.list \
    && apt-get update && apt-get install -y --no-install-recommends \
        ${EDGETPU_RUNTIME} \
        python3-tflite-runtime \
        python3-numpy \
        python3-opencv \
        python3-flask \
        python3-waitress \
        usbutils \
    && apt-get purge -y gnupg && apt-get autoremove -y \
    && rm -rf /var/lib/apt/lists/*

# Person-detection models (COCO; "person" = class 0), each as Edge TPU build + CPU fallback.
# Default is SSDLite MobileDet; the others can be chosen in the web UI (Settings → Model).
RUN mkdir -p /models && cd /models \
    && base=https://github.com/google-coral/test_data/raw/master \
    && for m in ssdlite_mobiledet_coco_qat_postprocess \
                ssd_mobilenet_v2_coco_quant_postprocess \
                efficientdet_lite0_320_ptq efficientdet_lite1_384_ptq \
                efficientdet_lite2_448_ptq efficientdet_lite3_512_ptq \
                efficientdet_lite3x_640_ptq; do \
         curl -fsSLO $base/${m}_edgetpu.tflite && curl -fsSLO $base/${m}.tflite || exit 1; \
       done \
    && curl -fsSLO $base/coco_labels.txt

COPY --from=ffmpeg /ffmpeg /ffprobe /usr/local/bin/
COPY --from=topdown /out/ /models/

WORKDIR /app
COPY app/ /app/
COPY scripts/ /scripts/

VOLUME ["/data"]
EXPOSE 8080
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s \
    CMD curl -fs http://localhost:${PORT}/healthz || exit 1

CMD ["python3", "/app/main.py"]
