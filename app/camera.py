"""Camera sources: USB webcams (V4L2), RTSP / RTSPS / HTTP streams via ffmpeg, Hikvision helper.

Every reader runs its own background thread and only keeps the newest frame, so a slow
detector never builds up latency on a live stream.
"""
import glob
import json
import logging
import os
import re
import subprocess
import threading
import time
from urllib.parse import quote

import cv2
import numpy as np

log = logging.getLogger("camera")

_CRED_RE = re.compile(r"^([a-zA-Z][a-zA-Z0-9+.-]*://)(.*)@([^@/]+)(/.*)?$")
HIK_STREAMS = {"main": "01", "sub": "02", "third": "03"}


# ---------------------------------------------------------------- config
def config_from_env():
    """Build the camera config from environment variables."""
    if os.environ.get("HIK_HOST"):
        return {
            "type": "hikvision",
            "hik_host": os.environ["HIK_HOST"],
            "hik_user": os.environ.get("HIK_USER", "admin"),
            "hik_pass": os.environ.get("HIK_PASS", ""),
            "hik_channel": int(os.environ.get("HIK_CHANNEL", "1")),
            "hik_stream": os.environ.get("HIK_STREAM", "main"),
            "hik_tls": os.environ.get("HIK_TLS", "0") == "1",
            "hik_port": int(os.environ["HIK_PORT"]) if os.environ.get("HIK_PORT") else None,
            "transport": os.environ.get("RTSP_TRANSPORT", "tcp"),
        }
    src = os.environ.get("SOURCE", "0").strip()
    if src.lower() == "demo":
        return {"type": "demo"}
    if src.isdigit() or src.startswith("/dev/"):
        return {"type": "usb", "device": src}
    return {"type": "url", "url": src, "transport": os.environ.get("RTSP_TRANSPORT", "tcp")}


def hikvision_url(cfg):
    tls = bool(cfg.get("hik_tls"))
    port = cfg.get("hik_port") or (322 if tls else 554)
    user = quote(str(cfg.get("hik_user") or ""), safe="")
    pw = quote(str(cfg.get("hik_pass") or ""), safe="")
    auth = f"{user}:{pw}@" if user else ""
    stream = HIK_STREAMS.get(cfg.get("hik_stream", "main"), "01")
    channel = int(cfg.get("hik_channel") or 1)
    scheme = "rtsps" if tls else "rtsp"
    return f"{scheme}://{auth}{cfg['hik_host']}:{port}/Streaming/Channels/{channel}{stream}"


def stream_url(cfg):
    if cfg["type"] == "hikvision":
        return hikvision_url(cfg)
    return cfg.get("url", "")


def fix_url_credentials(url):
    """Percent-encode user/password so passwords containing @ : / # ? work as typed."""
    m = _CRED_RE.match(url)
    if not m:
        return url
    scheme, creds, host, path = m.group(1), m.group(2), m.group(3), m.group(4) or ""
    user, _, pw = creds.partition(":")
    enc = lambda s: quote(s, safe="%")  # keep already-encoded sequences
    return f"{scheme}{enc(user)}:{enc(pw)}@{host}{path}" if _ else f"{scheme}{enc(user)}@{host}{path}"


def mask_url(url):
    """Hide the password, even when it was typed unencoded and contains @ : / #."""
    m = _CRED_RE.match(url or "")
    if not m:
        return url
    user = m.group(2).split(":", 1)[0]
    return f"{m.group(1)}{user}:\u2022\u2022\u2022@{m.group(3)}{m.group(4) or ''}"


def describe(cfg):
    t = cfg.get("type")
    if t == "usb":
        return f"USB {cfg.get('device')}"
    if t == "demo":
        return "Demo"
    return mask_url(stream_url(cfg))


def list_usb_devices():
    """Video capture nodes with their driver-reported names."""
    out = []
    for path in sorted(glob.glob("/dev/video*"), key=lambda p: int(re.sub(r"\D", "", p) or 0)):
        n = os.path.basename(path)
        name = _read(f"/sys/class/video4linux/{n}/name") or n
        index = _read(f"/sys/class/video4linux/{n}/index")
        if index not in (None, "0"):
            continue  # metadata nodes (video1, video3…) can't capture
        out.append({"device": path, "name": name})
    for path in sorted(glob.glob("/dev/v4l/by-id/*-video-index0")):
        out.append({"device": path, "name": os.path.basename(path), "stable": True})
    return out


def _read(path):
    try:
        with open(path) as f:
            return f.read().strip()
    except OSError:
        return None


# ---------------------------------------------------------------- readers
class _LatestFrame:
    def __init__(self):
        self._frame = None
        self._seq = 0
        self._cond = threading.Condition()
        self.error = None
        self.alive = True

    def put(self, frame):
        with self._cond:
            self._frame = frame
            self._seq += 1
            self._cond.notify_all()

    def get(self, last_seq, timeout):
        with self._cond:
            self._cond.wait_for(lambda: self._seq != last_seq or not self.alive, timeout=timeout)
            return self._seq, (self._frame if self._seq != last_seq else None)

    def stop(self):
        with self._cond:
            self.alive = False
            self._cond.notify_all()


class BaseReader:
    stall_timeout = 15.0

    def __init__(self):
        self.buf = _LatestFrame()
        self._seq = 0
        self.width = self.height = 0

    def read(self):
        """Return the newest unseen frame (BGR) or None if the stream stalled/died."""
        seq, frame = self.buf.get(self._seq, self.stall_timeout)
        if frame is None:
            return None
        self._seq = seq
        return frame

    @property
    def error(self):
        return self.buf.error

    def close(self):
        self.buf.stop()


class UsbReader(BaseReader):
    def __init__(self, device, width=None, height=None, fps=None, fourcc="MJPG"):
        super().__init__()
        dev = int(device) if str(device).isdigit() else device
        self.cap = cv2.VideoCapture(dev, cv2.CAP_V4L2)
        if not self.cap.isOpened():
            raise RuntimeError(f"cannot open USB camera {device} (is it passed to the container?)")
        if fourcc:  # MJPG gives most webcams far higher fps at 720p/1080p than YUYV
            self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*fourcc))
        if width:
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(width))
        if height:
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(height))
        if fps:
            self.cap.set(cv2.CAP_PROP_FPS, float(fps))
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        log.info("USB camera %s opened at %dx%d", device, self.width, self.height)
        threading.Thread(target=self._run, daemon=True).start()

    def _run(self):
        fails = 0
        while self.buf.alive:
            ok, frame = self.cap.read()
            if not ok:
                fails += 1
                if fails > 30:
                    self.buf.error = "USB camera stopped delivering frames"
                    break
                time.sleep(0.05)
                continue
            fails = 0
            self.buf.put(frame)
        self.cap.release()
        self.buf.stop()


class FFmpegReader(BaseReader):
    """Decodes any ffmpeg-supported URL (rtsp, rtsps, http(s) mjpeg/hls, files) to raw BGR."""

    def __init__(self, url, transport="tcp", max_width=1280, fps=None):
        super().__init__()
        self.url = fix_url_credentials(url)
        self.transport = transport if transport in ("tcp", "udp") else "tcp"
        self.input_args = self._input_args()
        w, h = self._probe()
        if w > max_width:
            h = int(round(h * max_width / w / 2) * 2)
            w = max_width
        self.width, self.height = w, h
        vf = f"scale={w}:{h}"
        if fps:
            vf = f"fps={fps}," + vf
        cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin", *self.input_args,
               "-i", self.url, "-an", "-sn", "-dn", "-vf", vf,
               "-pix_fmt", "bgr24", "-f", "rawvideo", "pipe:1"]
        self.proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                     bufsize=w * h * 3)
        log.info("streaming %s at %dx%d", mask_url(self.url), w, h)
        threading.Thread(target=self._run, daemon=True).start()
        threading.Thread(target=self._stderr, daemon=True).start()

    def _input_args(self):
        scheme = self.url.split(":", 1)[0].lower()
        args = []
        if scheme in ("rtsp", "rtsps"):
            # rtsps always runs interleaved over the TLS connection
            args += ["-rtsp_transport", "tcp" if scheme == "rtsps" else self.transport]
            args += ["-timeout", "10000000"]  # µs; drop dead connections
            args += ["-analyzeduration", "3000000", "-probesize", "3000000"]
        elif scheme in ("http", "https"):
            args += ["-timeout", "10000000", "-reconnect", "1", "-reconnect_streamed", "1"]
        elif os.path.exists(self.url):
            args += ["-re", "-stream_loop", "-1"]  # play files in real time, looping
        return args

    def _probe(self):
        cmd = ["ffprobe", "-v", "error", *self.input_args, "-select_streams", "v:0",
               "-show_entries", "stream=width,height", "-of", "json", self.url]
        try:
            out = subprocess.run(cmd, capture_output=True, text=True, timeout=25)
        except subprocess.TimeoutExpired:
            raise RuntimeError("timed out connecting to the camera")
        if out.returncode != 0:
            msg = (out.stderr.strip().splitlines() or ["unknown error"])[-1]
            raise RuntimeError(_friendly_error(msg.replace(self.url, mask_url(self.url))))
        streams = json.loads(out.stdout or "{}").get("streams") or []
        if not streams or not streams[0].get("width"):
            raise RuntimeError("stream has no video track")
        return int(streams[0]["width"]), int(streams[0]["height"])

    def _run(self):
        size = self.width * self.height * 3
        out = self.proc.stdout
        while self.buf.alive:
            data = bytearray(size)  # writable, so overlays can be drawn on it
            view, got = memoryview(data), 0
            while got < size:
                n = out.readinto(view[got:])
                if not n:
                    break
                got += n
            if got < size:
                break
            self.buf.put(np.frombuffer(data, np.uint8).reshape(self.height, self.width, 3))
        self.buf.error = self.buf.error or "stream ended"
        self.buf.stop()

    def _stderr(self):
        for line in self.proc.stderr:
            msg = line.decode(errors="replace").strip()
            if msg:
                self.buf.error = _friendly_error(msg.replace(self.url, mask_url(self.url)))
                log.warning("ffmpeg: %s", self.buf.error)

    def close(self):
        super().close()
        threading.Thread(target=self._reap, daemon=True).start()

    def _reap(self):
        if self.proc.poll() is None:
            self.proc.terminate()  # lets ffmpeg send RTSP TEARDOWN to the camera
            try:
                self.proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                self.proc.wait()


def _friendly_error(msg):
    low = msg.lower()
    if "401" in low or "unauthorized" in low:
        return "camera rejected the username/password (401 Unauthorized)"
    if "404" in low or "not found" in low:
        return "stream path not found (404) — check the channel / URL path"
    if "connection refused" in low:
        return "connection refused — wrong port, or RTSP/RTSPS disabled on the camera"
    if "timed out" in low or "timeout" in low:
        return "connection timed out — camera unreachable from the container"
    if "tls" in low or "ssl" in low or "handshake" in low:
        return f"TLS error ({msg}) — is RTSPS enabled on that port?"
    return msg


def open_reader(cfg, max_width=1280, fps=None):
    t = cfg.get("type")
    if t == "usb":
        return UsbReader(cfg.get("device", "0"),
                         cfg.get("width") or os.environ.get("CAMERA_WIDTH"),
                         cfg.get("height") or os.environ.get("CAMERA_HEIGHT"),
                         cfg.get("fps") or os.environ.get("CAMERA_FPS"),
                         os.environ.get("CAMERA_FOURCC", "MJPG"))
    if t in ("url", "hikvision"):
        url = stream_url(cfg)
        if not url:
            raise RuntimeError("no camera URL configured")
        return FFmpegReader(url, cfg.get("transport", "tcp"), max_width=max_width, fps=fps)
    raise RuntimeError(f"unknown camera type {t!r}")


def grab_test_frame(cfg):
    """Open the source, take one frame, close. Used by the 'Test' button."""
    reader = open_reader(cfg, max_width=960)
    try:
        reader.stall_timeout = 12
        frame = reader.read()
        if frame is None:
            raise RuntimeError(reader.error or "connected but no frames arrived")
        return frame, reader.width, reader.height
    finally:
        reader.close()
