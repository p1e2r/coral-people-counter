"""Person detection: Coral Edge TPU models (CPU fallback), detection zone, tiling, demo.

Accuracy features
- Better models than the classic SSD MobileNet v2 (MobileDet, EfficientDet-Lite).
- Detection zone: only the region you care about is sent to the model, so people
  occupy more of its input pixels.
- Aspect-correct tiling: a 16:9 frame squashed into a square input makes people thin
  and hard to recognise. The zone is split into near-square overlapping tiles instead,
  and detections are merged across tile borders.
"""
import logging
import math
import os
import random
import time

import numpy as np

log = logging.getLogger("detector")

MAX_ASPECT = float(os.environ.get("MAX_BOX_ASPECT", "1.8"))  # width / height
NESTED_THR = 0.9
PERSON_CLASS = 0  # "person" in the COCO label map used by every Coral detection model
MODEL_DIR = os.environ.get("MODEL_DIR", "/models")

# name: (file stem, label, note, kind, view)
MODELS = {
    "mobiledet": ("ssdlite_mobiledet_coco_qat_postprocess", "SSDLite MobileDet 320",
                  "fast, good (95% counting F1 in tests)", "ssd", "angled"),
    "efficientdet_lite0": ("efficientdet_lite0_320_ptq", "EfficientDet-Lite0 320", "accurate on large people", "ssd", "angled"),
    "efficientdet_lite1": ("efficientdet_lite1_384_ptq", "EfficientDet-Lite1 384",
                           "Recommended: most accurate in tests (31/31 crossings counted correctly)", "ssd", "angled"),
    "efficientdet_lite2": ("efficientdet_lite2_448_ptq", "EfficientDet-Lite2 448", "as accurate as Lite1, slower", "ssd", "angled"),
    "efficientdet_lite3": ("efficientdet_lite3_512_ptq", "EfficientDet-Lite3 512", "very precise, slow", "ssd", "angled"),
    "efficientdet_lite3x": ("efficientdet_lite3x_640_ptq", "EfficientDet-Lite3x 640", "slowest; best on still photos, worse on video", "ssd", "angled"),
    "ssd_mobilenet_v2": ("ssd_mobilenet_v2_coco_quant_postprocess", "SSD MobileNet v2 300", "fastest, least accurate", "ssd", "angled"),
    "topdown": ("topdown", "Top-down YOLOv8n (overhead people)",
                "trained on top-view people, UPCT 2025, CC BY 4.0", "yolo", "topdown"),
}
DEFAULT_MODEL = "efficientdet_lite1"
TOPDOWN_MODEL = "topdown"
# /data/models first, so you can drop in your own (re)trained files without rebuilding
MODEL_DIRS = [d for d in os.environ.get("MODEL_DIRS", "/data/models:" + MODEL_DIR).split(":") if d]


def find_model_file(name):
    for d in MODEL_DIRS:
        p = os.path.join(d, name)
        if os.path.exists(p):
            return p
    return None


def model_available(key):
    stem = MODELS[key][0]
    return bool(find_model_file(stem + "_edgetpu.tflite") or find_model_file(stem + ".tflite"))


def model_list(view=None):
    out = []
    keys = [DEFAULT_MODEL] + [k for k in MODELS if k != DEFAULT_MODEL]  # recommended first
    for key in keys:
        stem, label, note, kind, v = MODELS[key]
        if (view is None or v == view) and model_available(key):
            out.append({"id": key, "label": label, "note": note, "view": v})
    return out


def topdown_status():
    if model_available(TOPDOWN_MODEL):
        return {"available": True}
    for d in MODEL_DIRS:
        f = os.path.join(d, "TOPDOWN_FAILED.txt")
        if os.path.exists(f):
            with open(f) as fh:
                return {"available": False, "reason": fh.read().strip()}
    return {"available": False, "reason": "top-down model not built into this image"}


# ------------------------------------------------------------------ geometry helpers
def iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def overlap_min(a, b):
    """Intersection over the smaller box: catches one person split by a tile border."""
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    small = min((a[2] - a[0]) * (a[3] - a[1]), (b[2] - b[0]) * (b[3] - b[1]))
    return inter / small if small > 0 else 0.0


def merge_detections(dets, iou_thr=0.5, contain_thr=0.6):
    """Merge duplicates of the same person seen by two overlapping tiles.

    dets: [(box, score, tile_id, clipped)] where `clipped` means the box touches an inner
    tile border (the person continues in the neighbouring tile). Boxes from the SAME tile
    are never merged — the model already ran NMS there, and two overlapping boxes are
    usually two real people (one partly behind the other).
    """
    dets = sorted(dets, key=lambda d: d[1], reverse=True)
    kept = []  # [box, score, tiles:set, clipped]
    for box, score, tile, clipped in dets:
        for k in kept:
            if tile in k[2]:
                # same tile: only drop a lower-scored box lying almost entirely inside a
                # stronger one (a torso/half-body duplicate). Partly hidden people stick out
                # of the front person's box, so they stay.
                if overlap_min(box, k[0]) > NESTED_THR:
                    break
                continue
            same = iou(box, k[0]) > iou_thr or (
                (clipped or k[3]) and overlap_min(box, k[0]) > contain_thr)
            if same:
                k[0] = (min(box[0], k[0][0]), min(box[1], k[0][1]),
                        max(box[2], k[0][2]), max(box[3], k[0][3])) if (clipped or k[3]) else k[0]
                k[2].add(tile)
                k[3] = k[3] and clipped
                break
        else:
            kept.append([box, score, {tile}, clipped])
    return [(tuple(k[0]), k[1]) for k in kept]


def plan_tiles(zone, frame_w, frame_h, tiles, overlap=0.2):
    """Split the zone (normalised x1,y1,x2,y2) into a grid of overlapping, near-square tiles."""
    zx1, zy1, zx2, zy2 = zone
    zw, zh = (zx2 - zx1) * frame_w, (zy2 - zy1) * frame_h
    if tiles == "auto":
        ratio = zw / max(zh, 1)
        cols = max(1, min(4, round(ratio))) if ratio >= 1 else 1
        rows = max(1, min(4, round(1 / ratio))) if ratio < 1 else 1
    else:
        n = int(tiles)
        cols, rows = {1: (1, 1), 2: (2, 1), 3: (3, 1), 4: (2, 2), 6: (3, 2), 9: (3, 3)}.get(n, (1, 1))
        if zh > zw and cols > rows:
            cols, rows = rows, cols
    out = []
    tw = (zx2 - zx1) / (cols - (cols - 1) * overlap)
    th = (zy2 - zy1) / (rows - (rows - 1) * overlap)
    for r in range(rows):
        for c in range(cols):
            x1 = zx1 + c * tw * (1 - overlap)
            y1 = zy1 + r * th * (1 - overlap)
            out.append((x1, y1, min(zx2, x1 + tw), min(zy2, y1 + th)))
    return out


# ------------------------------------------------------------------ detectors
class TFLiteDetector:
    """Runs an SSD/EfficientDet detection model through tflite_runtime (+ Edge TPU)."""

    def __init__(self, model_key=DEFAULT_MODEL, device="usb", allow_cpu=True,
                 tpu_path=None, cpu_path=None):
        try:
            from tflite_runtime.interpreter import Interpreter, load_delegate
        except ImportError:  # newer name of the same runtime
            from ai_edge_litert.interpreter import Interpreter, load_delegate

        if model_key not in MODELS:
            model_key = DEFAULT_MODEL
        stem, label, _, self.kind, self.view = MODELS[model_key]
        tpu_path = tpu_path or find_model_file(stem + "_edgetpu.tflite") or "/missing"
        cpu_path = cpu_path or find_model_file(stem + ".tflite") or "/missing"
        self.model_key, self.model_label = model_key, label
        try:
            opts = {"device": device} if device else {}
            delegate = load_delegate("libedgetpu.so.1", opts)
            self.interp = Interpreter(model_path=tpu_path, experimental_delegates=[delegate])
            self.backend = f"Coral Edge TPU · {label}"
        except Exception as e:  # no Coral attached / permission problem
            if not allow_cpu:
                raise RuntimeError(f"Could not open Coral Edge TPU: {e}") from e
            log.warning("Edge TPU unavailable (%s) — falling back to CPU", e)
            self.interp = Interpreter(model_path=cpu_path, num_threads=4)
            self.backend = f"CPU (no Coral) · {label}"
        self.interp.allocate_tensors()
        inp = self.interp.get_input_details()[0]
        self.input_index = inp["index"]
        self.input_dtype = inp["dtype"]
        self.input_quant = inp.get("quantization", (0.0, 0))
        _, self.h, self.w, _ = inp["shape"]
        self._outputs = self.interp.get_output_details()
        self._map = None
        log.info("Detector ready: %s, input %dx%d", self.backend, self.w, self.h)

    def _parse(self):
        outs = [self.interp.get_tensor(o["index"]) for o in self._outputs]
        if self._map is None:
            # boxes: [1,N,4]; count: [1]; classes vs scores: classes are whole numbers.
            m = {}
            two = []
            for i, o in enumerate(outs):
                if o.ndim == 3 and o.shape[-1] == 4:
                    m["boxes"] = i
                elif o.size == 1:
                    m["count"] = i
                else:
                    two.append(i)
            a, b = two
            a_int = np.all(np.mod(outs[a], 1) == 0)
            b_int = np.all(np.mod(outs[b], 1) == 0)
            if a_int != b_int:
                m["classes"], m["scores"] = (a, b) if a_int else (b, a)
                self._map = m  # only cache once the assignment is unambiguous
            else:
                m["classes"], m["scores"] = a, b
            mapping = m
        else:
            mapping = self._map
        count = int(outs[mapping["count"]].flatten()[0])
        return (outs[mapping["boxes"]][0][:count], outs[mapping["classes"]][0][:count],
                outs[mapping["scores"]][0][:count])

    def _infer(self, img_rgb, threshold):
        import cv2

        img = cv2.resize(img_rgb, (self.w, self.h), interpolation=cv2.INTER_AREA
                         if img_rgb.shape[1] > self.w else cv2.INTER_LINEAR)
        if self.kind == "yolo":
            return self._infer_yolo(img, threshold)
        if self.input_dtype != np.uint8:
            img = (img.astype(np.float32) - 127.5) / 127.5
        self.interp.set_tensor(self.input_index, np.expand_dims(img, 0).astype(self.input_dtype))
        self.interp.invoke()
        boxes, classes, scores = self._parse()
        found = []
        for box, cls, score in zip(boxes, classes, scores):
            if int(cls) == PERSON_CLASS and score >= threshold:
                ymin, xmin, ymax, xmax = [float(np.clip(v, 0, 1)) for v in box]
                if xmax > xmin and ymax > ymin:
                    found.append(((xmin, ymin, xmax, ymax), float(score)))
        return found

    def _infer_yolo(self, img, threshold):
        """Ultralytics YOLO exported with format=edgetpu: one output [1, 4+classes, N]
        holding normalised centre-x, centre-y, width, height and per-class scores."""
        import cv2

        x = img.astype(np.float32) / 255.0
        scale, zp = self.input_quant
        if self.input_dtype in (np.int8, np.uint8) and scale:
            info = np.iinfo(self.input_dtype)
            x = np.clip(np.round(x / scale + zp), info.min, info.max)
        self.interp.set_tensor(self.input_index, np.expand_dims(x, 0).astype(self.input_dtype))
        self.interp.invoke()
        o = self._outputs[0]
        y = self.interp.get_tensor(o["index"]).astype(np.float32)[0]
        oscale, ozp = o.get("quantization", (0.0, 0))
        if oscale:
            y = (y - ozp) * oscale
        if y.shape[0] > y.shape[1]:
            y = y.T                      # -> [4+classes, N]
        xywh, scores = y[:4], y[4]       # class 0 = person
        if xywh.max() > 2.0:             # exported without normalisation: pixel units
            xywh = xywh / np.array([self.w, self.h, self.w, self.h], np.float32)[:, None]
        keep = np.where(scores >= threshold)[0]
        if keep.size == 0:
            return []
        cx, cy, bw, bh = xywh[:, keep]
        sc = scores[keep]
        rects = [[float(a - c / 2), float(b - d / 2), float(c), float(d)] for a, b, c, d in zip(cx, cy, bw, bh)]
        idx = cv2.dnn.NMSBoxes(rects, sc.tolist(), float(threshold), 0.5)
        found = []
        for i in np.array(idx).flatten():
            x1, y1, w, h = rects[i]
            box = (max(0.0, x1), max(0.0, y1), min(1.0, x1 + w), min(1.0, y1 + h))
            if box[2] > box[0] and box[3] > box[1]:
                found.append((box, float(sc[i])))
        return found

    def detect(self, frame_rgb, threshold, zone=(0, 0, 1, 1), tiles="auto"):
        """Return [(box, score)] in normalised full-frame coords, score >= threshold."""
        h, w = frame_rgb.shape[:2]
        dets = []
        grid = plan_tiles(zone, w, h, tiles)
        zx1, zy1, zx2, zy2 = zone
        edge = 0.01  # tile-relative distance that counts as "touching the border"
        for tid, (tx1, ty1, tx2, ty2) in enumerate(grid):
            px1, py1, px2, py2 = int(tx1 * w), int(ty1 * h), int(math.ceil(tx2 * w)), int(math.ceil(ty2 * h))
            crop = frame_rgb[py1:py2, px1:px2]
            if crop.size == 0:
                continue
            sx, sy = (px2 - px1) / w, (py2 - py1) / h
            ox, oy = px1 / w, py1 / h
            inner = (tx1 > zx1 + 1e-6, ty1 > zy1 + 1e-6, tx2 < zx2 - 1e-6, ty2 < zy2 - 1e-6)
            for (x1, y1, x2, y2), s in self._infer(crop, threshold):
                clipped = (inner[0] and x1 < edge) or (inner[1] and y1 < edge) or \
                          (inner[2] and x2 > 1 - edge) or (inner[3] and y2 > 1 - edge)
                dets.append(((ox + x1 * sx, oy + y1 * sy, ox + x2 * sx, oy + y2 * sy), s, tid, clipped))
        people = []
        for box, score in merge_detections(dets):
            bw, bh = (box[2] - box[0]) * w, (box[3] - box[1]) * h
            max_aspect = MAX_ASPECT if self.view == "angled" else 3.0
            if bh <= 0 or bw <= 0 or bw / bh > max_aspect or (self.view == "topdown" and bh / bw > 3.0):
                continue  # far wider than tall: not a person (horizon lines, cars, benches…)
            people.append((box, score))
        return people


class DemoDetector:
    """Generates synthetic people walking across a fake scene, for testing the UI."""

    backend = "Demo (synthetic)"
    view = "angled"

    def __init__(self, width=960, height=540):
        self.w, self.h = width, height
        self.people = []
        self.last = time.time()
        self._dets = []

    def frame(self):
        img = np.zeros((self.h, self.w, 3), dtype=np.uint8)
        img[:] = (28, 32, 40)
        img[int(self.h * 0.62):, :] = (44, 48, 58)
        now = time.time()
        dt, self.last = now - self.last, now
        if random.random() < 0.02 and len(self.people) < 6:
            ltr = random.random() < 0.5
            self.people.append({
                "x": -0.08 if ltr else 1.08,
                "y": random.uniform(0.35, 0.65),
                "v": random.uniform(0.08, 0.2) * (1 if ltr else -1),
                "phase": random.uniform(0, 6.28),
            })
        for p in self.people:
            p["x"] += p["v"] * dt
            p["y"] += math.sin(now * 2 + p["phase"]) * 0.002
        self.people = [p for p in self.people if -0.15 < p["x"] < 1.15]
        dets = []
        for p in self.people:
            bw, bh = 0.07, 0.3
            box = (p["x"] - bw / 2, p["y"] - bh / 2, p["x"] + bw / 2, p["y"] + bh / 2)
            x1, y1, x2, y2 = [int(v * s) for v, s in zip(box, (self.w, self.h, self.w, self.h))]
            img[max(0, y1):max(0, y2), max(0, x1):max(0, x2)] = (120, 150, 190)
            if 0 <= box[0] and box[2] <= 1:
                dets.append((box, 0.9))
        self._dets = dets
        return img

    def detect(self, frame_rgb, threshold, zone=(0, 0, 1, 1), tiles="auto"):
        return [d for d in self._dets if d[1] >= threshold]
