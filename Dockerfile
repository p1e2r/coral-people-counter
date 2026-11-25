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

# Coral People Counter — runtime image
# Google retired its Coral apt repo, so the Edge TPU runtime comes from feranick's
# maintained builds (as used by Frigate): libedgetpu 16.0 built against TF 2.17.1
# and the matching tflite_runtime 2.17.1 wheel, for Debian bookworm / Python 3.11.
FROM debian:bookworm-slim

ARG TARGETARCH
ARG EDGETPU_RUNTIME=std
# EDGETPU_RUNTIME=max for full clock speed (faster, runs hotter)
ARG LIBEDGETPU_VERSION=16.0TF2.17.1-1
ARG TFLITE_VERSION=2.17.1

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    TZ=UTC \
    DATA_DIR=/data \
    PORT=8080

RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates curl tzdata usbutils libusb-1.0-0 \
        python3 python3-pip python3-numpy python3-opencv python3-flask python3-waitress \
    && case "$TARGETARCH" in \
         amd64) DEB_ARCH=amd64; WHL_ARCH=x86_64 ;; \
         arm64) DEB_ARCH=arm64; WHL_ARCH=aarch64 ;; \
         *) echo "unsupported architecture: $TARGETARCH" >&2; exit 1 ;; \
       esac \
    && ver_lc=$(echo "$LIBEDGETPU_VERSION" | sed 's/TF/tf/') \
    && cd /tmp \
    && curl -fsSLO "https://github.com/feranick/libedgetpu/releases/download/${LIBEDGETPU_VERSION}/libedgetpu1-${EDGETPU_RUNTIME}_${ver_lc}.bookworm_${DEB_ARCH}.deb" \
    && apt-get install -y --no-install-recommends ./libedgetpu1-*.deb \
    && curl -fsSLO "https://github.com/feranick/TFlite-builds/releases/download/v${TFLITE_VERSION}/tflite_runtime-${TFLITE_VERSION}-cp311-cp311-linux_${WHL_ARCH}.whl" \
    && pip3 install --no-cache-dir --break-system-packages ./tflite_runtime-*.whl \
    && python3 -c "import tflite_runtime.interpreter, cv2, flask, waitress, numpy; print('runtime ok', numpy.__version__, cv2.__version__)" \
    && rm -rf /tmp/*.deb /tmp/*.whl /var/lib/apt/lists/*

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
