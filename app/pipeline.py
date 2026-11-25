"""Camera -> detector -> tracker -> counter loop, running in a background thread."""
import logging
import os
import threading
import time

import cv2
import numpy as np

import camera
import detector as det_mod
from detector import DemoDetector, TFLiteDetector
from tracker import Counter, Tracker

log = logging.getLogger("pipeline")
PASSWORD_MASK = "\u2022\u2022\u2022\u2022\u2022\u2022"

DEFAULTS = {
    "mode": "line",            # line | unique
    "line": [0.5, 0.05, 0.5, 0.95],
    "threshold": 0.45,         # confident detections; ByteTrack also uses ones down to ~half this
    "min_hits": 3,
    "count_dirs": "both",      # both | in | out
    "invert": False,
    "view": "angled",          # angled | topdown — camera mounting, picks the model family
    "model": det_mod.DEFAULT_MODEL,  # model used for the angled view
    "tiles": "auto",           # auto | 1 | 2 | 3 | 4 | 6
    "zone": [0.0, 0.0, 1.0, 1.0],  # detection zone x1,y1,x2,y2
    "anchor": "center",        # center | bottom | top — point that must cross the line
    "snapshots": True,
}
SNAPSHOT_KEEP = int(os.environ.get("SNAPSHOT_KEEP", "500"))


def _env_float(name, default):
    try:
        return float(os.environ.get(name, default))
    except ValueError:
        return default


def default_settings():
    s = dict(DEFAULTS)
    if os.environ.get("COUNT_MODE"):
        s["mode"] = os.environ["COUNT_MODE"]
    if os.environ.get("COUNT_LINE"):
        s["line"] = [float(v) for v in os.environ["COUNT_LINE"].split(",")]
    s["threshold"] = _env_float("THRESHOLD", s["threshold"])
    if os.environ.get("MODEL"):
        s["model"] = os.environ["MODEL"]
    if os.environ.get("CAMERA_VIEW") in ("angled", "topdown"):
        s["view"] = os.environ["CAMERA_VIEW"]
    if os.environ.get("TILES"):
        s["tiles"] = os.environ["TILES"]
    if os.environ.get("ANCHOR"):
        s["anchor"] = os.environ["ANCHOR"]
    return s


class Pipeline(threading.Thread):
    def __init__(self, store):
        super().__init__(daemon=True)
        self.store = store
        self.process_width = int(os.environ.get("PROCESS_WIDTH", "1280"))
        self.max_fps = _env_float("MAX_FPS", 30)
        self.stream_width = int(os.environ.get("STREAM_WIDTH", "960"))
        self.settings = default_settings()
        self.settings.update({k: v for k, v in store.get_settings().items() if k != "camera"})
        self.counter = Counter(self._on_event)
        self.counter.configure(dict(self.settings, anchor=self.effective_anchor()))
        self.tracker = Tracker(min_hits=self.settings["min_hits"])
        self.snap_dir = os.path.join(os.environ.get("DATA_DIR", "/data"), "snapshots")
        os.makedirs(self.snap_dir, exist_ok=True)
        self._frame = None
        self.fps = 0.0
        self.inference_ms = 0.0
        self.status = "starting"
        self.backend = "-"
        self.active = 0
        self._jpeg = None
        self._jpeg_cond = threading.Condition()
        self._lock = threading.Lock()

        self._restart = threading.Event()
        self._demo = DemoDetector()
        self._tflite = None
        self.detector = self._demo
        self.source_desc = "-"
        saved = store.get_settings().get("camera")
        self.camera = saved if saved and os.environ.get("CAMERA_FROM_ENV") != "1" \
            else camera.config_from_env()

    # ---------- settings ----------
    def update_settings(self, values):
        clean = {}
        if values.get("mode") in ("line", "unique"):
            clean["mode"] = values["mode"]
        if "line" in values:
            line = [min(1.0, max(0.0, float(v))) for v in values["line"]][:4]
            if len(line) == 4:
                clean["line"] = line
        if "threshold" in values:
            clean["threshold"] = min(0.95, max(0.1, float(values["threshold"])))
        if "min_hits" in values:
            clean["min_hits"] = max(1, min(30, int(values["min_hits"])))
        if values.get("count_dirs") in ("both", "in", "out"):
            clean["count_dirs"] = values["count_dirs"]
        if "invert" in values:
            clean["invert"] = bool(values["invert"])
        if values.get("model") in det_mod.MODELS and det_mod.MODELS[values["model"]][4] == "angled":
            clean["model"] = values["model"]
        if values.get("view") in ("angled", "topdown"):
            clean["view"] = values["view"]
        if str(values.get("tiles")) in ("auto", "1", "2", "3", "4", "6"):
            clean["tiles"] = str(values["tiles"])
        if "zone" in values:
            z = [min(1.0, max(0.0, float(v))) for v in values["zone"]][:4]
            if len(z) == 4:
                z = [min(z[0], z[2]), min(z[1], z[3]), max(z[0], z[2]), max(z[1], z[3])]
                if z[2] - z[0] > 0.05 and z[3] - z[1] > 0.05:
                    clean["zone"] = z
        if values.get("anchor") in ("center", "bottom", "top"):
            clean["anchor"] = values["anchor"]
        if "snapshots" in values:
            clean["snapshots"] = bool(values["snapshots"])
        with self._lock:
            before = self.effective_model()
            self.settings.update(clean)
            model_changed = self.effective_model() != before
            self.counter.configure(dict(self.settings, anchor=self.effective_anchor()))
            self.tracker.min_hits = self.settings["min_hits"]
        self.store.save_settings(clean)
        if model_changed:
            self._tflite = None
            self._restart.set()
        return dict(self.settings)

    def effective_model(self):
        """Model actually used: the top-down model when the view is top-down (if installed)."""
        if self.settings.get("view") == "topdown" and det_mod.model_available(det_mod.TOPDOWN_MODEL):
            return det_mod.TOPDOWN_MODEL
        return self.settings.get("model", det_mod.DEFAULT_MODEL)

    def view_warning(self):
        if self.settings.get("view") != "topdown":
            return None
        st = det_mod.topdown_status()
        if st["available"]:
            return None
        return ("Top-down model unavailable (" + st["reason"] + ") — using the angled model, "
                "which is unreliable straight down. Rebuild the image with network access to zenodo.org.")

    def effective_anchor(self):
        # from above, the box centre is the person's position on the floor
        return "center" if self.settings.get("view") == "topdown" else self.settings.get("anchor", "center")

    # ---------- counting events ----------
    def _on_event(self, direction, track):
        if direction == "undo":
            if track.event_id:
                self.store.delete_event(track.event_id)
                self._remove_snapshot(track.event_id)
                track.event_id = None
            return
        track.event_id = self.store.add_event(direction)
        if self.settings.get("snapshots") and self._frame is not None:
            self._save_snapshot(track.event_id, self._frame, track, direction)

    def _save_snapshot(self, event_id, frame, track, direction):
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = track.box
        bw, bh = x2 - x1, y2 - y1
        cx1, cy1 = max(0, int((x1 - bw * 0.6) * w)), max(0, int((y1 - bh * 0.25) * h))
        cx2, cy2 = min(w, int((x2 + bw * 0.6) * w)), min(h, int((y2 + bh * 0.25) * h))
        if cx2 - cx1 < 8 or cy2 - cy1 < 8:
            return
        crop = frame[cy1:cy2, cx1:cx2].copy()
        cv2.rectangle(crop, (int(x1 * w) - cx1, int(y1 * h) - cy1), (int(x2 * w) - cx1, int(y2 * h) - cy1),
                      (31, 90, 255), 2)
        scale = 200 / crop.shape[0]
        crop = cv2.resize(crop, (max(1, int(crop.shape[1] * scale)), 200))
        cv2.imwrite(os.path.join(self.snap_dir, f"{event_id}.jpg"), crop, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if event_id % 50 == 0:
            self._prune_snapshots()

    def _remove_snapshot(self, event_id):
        try:
            os.remove(os.path.join(self.snap_dir, f"{event_id}.jpg"))
        except OSError:
            pass

    def _prune_snapshots(self):
        files = sorted((int(f[:-4]) for f in os.listdir(self.snap_dir) if f.endswith(".jpg") and f[:-4].isdigit()))
        for eid in files[:-SNAPSHOT_KEEP]:
            self._remove_snapshot(eid)

    # ---------- camera ----------
    def camera_config(self, masked=False):
        cfg = dict(self.camera)
        if masked and cfg.get("hik_pass"):
            cfg["hik_pass"] = PASSWORD_MASK
        if masked and cfg.get("url"):
            cfg["url"] = camera.mask_url(cfg["url"])
        return cfg

    def merge_camera(self, cfg):
        """Fill masked secrets back in from the current config."""
        cfg = dict(cfg)
        if cfg.get("hik_pass") == PASSWORD_MASK:
            cfg["hik_pass"] = self.camera.get("hik_pass", "")
        if cfg.get("url") and "\u2022" in cfg["url"]:
            cfg["url"] = self.camera.get("url", "")
        return cfg

    def set_camera(self, cfg):
        cfg = self.merge_camera(cfg)
        if cfg.get("type") not in ("usb", "url", "hikvision", "demo"):
            raise ValueError("type must be usb, url, hikvision or demo")
        self.camera = cfg
        self.store.save_settings({"camera": cfg})
        self._restart.set()
        return self.camera_config(masked=True)

    def run(self):
        while True:
            self._restart.clear()
            cfg = dict(self.camera)
            demo = cfg.get("type") == "demo"
            self.detector = self._demo if demo else self._real_detector()
            self.backend = self.detector.backend
            self.source_desc = camera.describe(cfg)
            reader = None
            if not demo:
                self.status = f"connecting to {self.source_desc}"
                try:
                    reader = camera.open_reader(cfg, max_width=self.process_width,
                                                fps=self.max_fps or None)
                except Exception as e:
                    self.status = f"camera error: {e}"
                    self.fps = 0.0
                    self.active = 0
                    log.warning("%s (%s) — retrying in 10s", self.status, self.source_desc)
                    self._restart.wait(10)
                    continue
            self.status = "running"
            self.tracker = Tracker(min_hits=self.settings["min_hits"])
            try:
                self._loop(reader)
            except Exception:
                log.exception("pipeline crashed, restarting")
                self.status = "error, restarting"
            finally:
                if reader is not None:
                    reader.close()
            if not self._restart.is_set():
                self._restart.wait(3)

    def _real_detector(self):
        if self._tflite is None:
            self._tflite = TFLiteDetector(
                self.effective_model(),
                device=os.environ.get("EDGETPU_DEVICE", "usb"),
                allow_cpu=os.environ.get("ALLOW_CPU_FALLBACK", "1") == "1",
            )
        return self._tflite

    def _loop(self, reader):
        frame_interval = 1.0 / self.max_fps if self.max_fps > 0 else 0
        last_tick, frames, fps_t0 = time.time(), 0, time.time()
        while not self._restart.is_set():
            if reader is None:
                frame = cv2.cvtColor(self.detector.frame(), cv2.COLOR_RGB2BGR)
            else:
                frame = reader.read()
                if frame is None:
                    self.status = f"camera lost: {reader.error or 'no frames'} — reconnecting"
                    self.fps = 0.0
                    log.warning(self.status)
                    return
            self.status = "running"
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            with self._lock:
                threshold = self.settings["threshold"]
                zone, tiles = tuple(self.settings["zone"]), self.settings["tiles"]
            low = max(0.15, threshold * 0.5)
            t0 = time.perf_counter()
            dets = self.detector.detect(rgb, low, zone=zone, tiles=tiles)
            self.inference_ms = self.inference_ms * 0.9 + (time.perf_counter() - t0) * 100
            # keep tracks alive ~1.5 s while hidden, whatever the frame rate
            self.tracker.max_missed = max(8, int((self.fps or self.max_fps or 15) * 1.5))
            tracks = self.tracker.update(dets, threshold)
            self._frame = frame
            with self._lock:
                self.counter.process(tracks)
            self.active = sum(1 for t in tracks if t.missed == 0 and t.confirmed)
            self._publish(frame, tracks)

            frames += 1
            now = time.time()
            if now - fps_t0 >= 1.0:
                self.fps = frames / (now - fps_t0)
                frames, fps_t0 = 0, now
            sleep = frame_interval - (now - last_tick)
            if sleep > 0:
                time.sleep(sleep)
            last_tick = time.time()

    # ---------- preview stream ----------
    def _publish(self, frame, tracks):
        h, w = frame.shape[:2]
        if w > self.stream_width:
            scale = self.stream_width / w
            frame = cv2.resize(frame, (self.stream_width, int(h * scale)))
            h, w = frame.shape[:2]
        with self._lock:
            mode, line = self.settings["mode"], self.settings["line"]
            zone, anchor = self.settings["zone"], self.effective_anchor()
        if tuple(zone) != (0.0, 0.0, 1.0, 1.0):
            zx1, zy1, zx2, zy2 = int(zone[0] * w), int(zone[1] * h), int(zone[2] * w), int(zone[3] * h)
            shade = frame.copy()
            shade[:] = (shade * 0.45).astype(np.uint8)
            shade[zy1:zy2, zx1:zx2] = frame[zy1:zy2, zx1:zx2]
            frame = shade
            cv2.rectangle(frame, (zx1, zy1), (zx2, zy2), (255, 200, 60), 1, cv2.LINE_AA)
        if mode == "line":
            p1 = (int(line[0] * w), int(line[1] * h))
            p2 = (int(line[2] * w), int(line[3] * h))
            cv2.line(frame, p1, p2, (31, 90, 255), 3, cv2.LINE_AA)
        for t in tracks:
            if not t.confirmed or t.missed > 2:
                continue
            color = (120, 220, 80) if t.missed == 0 else (120, 160, 160)
            x1, y1, x2, y2 = (int(t.box[0] * w), int(t.box[1] * h),
                              int(t.box[2] * w), int(t.box[3] * h))
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            label = f"#{t.id} {int(t.score * 100)}%"
            cv2.putText(frame, label, (x1, max(14, y1 - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
            ax, ay = t.anchor(anchor)
            cv2.circle(frame, (int(ax * w), int(ay * h)), 4, (31, 90, 255) if t.counted_at else color, -1)
            if len(t.trail) > 1:
                pts = np.array([(int(px * w), int(py * h)) for px, py in t.trail], np.int32)
                cv2.polylines(frame, [pts], False, color, 1, cv2.LINE_AA)
        ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
        if ok:
            with self._jpeg_cond:
                self._jpeg = buf.tobytes()
                self._jpeg_cond.notify_all()

    def frames(self):
        while True:
            with self._jpeg_cond:
                self._jpeg_cond.wait(timeout=2)
                jpg = self._jpeg
            if jpg:
                yield (b"--frame\r\nContent-Type: image/jpeg\r\n"
                       b"Content-Length: " + str(len(jpg)).encode() + b"\r\n\r\n" + jpg + b"\r\n")

    def snapshot(self):
        return self._jpeg
