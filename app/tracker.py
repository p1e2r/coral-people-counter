"""ByteTrack-style multi-person tracker with Kalman motion model, plus line-crossing counter.

Why this matters for counting accuracy
- Kalman prediction: a track is matched against where the person *should* be now, so
  fast walkers and brief occlusions (someone walking behind another) keep their ID.
- Two-stage association (ByteTrack): after matching confident detections, leftover
  tracks are matched to LOW-confidence detections. A partly hidden person often drops
  to a low score for a few frames; using those boxes keeps the track alive instead of
  losing it and later re-counting the same person as new.
- New tracks start only from confident detections and must be confirmed over several
  frames, so a one-frame false positive can never be counted.
"""
import itertools
import time

import numpy as np


def iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


# ---------------------------------------------------------------- Kalman filter
class KalmanBox:
    """Constant-velocity model on (cx, cy, w, h)."""

    _F = np.eye(8)
    _F[:4, 4:] = np.eye(4)
    _H = np.eye(4, 8)

    def __init__(self, box):
        cx, cy, w, h = self._to_xywh(box)
        self.x = np.array([cx, cy, w, h, 0, 0, 0, 0], dtype=float)
        s = max(h, 1e-3)
        self.P = np.diag([2 * s, 2 * s, 2 * s, 2 * s, 10 * s, 10 * s, 10 * s, 10 * s]) ** 2 * 0.01

    @staticmethod
    def _to_xywh(b):
        return ((b[0] + b[2]) / 2, (b[1] + b[3]) / 2, b[2] - b[0], b[3] - b[1])

    def _noise(self, pos, vel):
        h = max(self.x[3], 1e-3)
        return np.diag([pos * h, pos * h, pos * h, pos * h, vel * h, vel * h, vel * h, vel * h]) ** 2

    def predict(self):
        self.x = self._F @ self.x
        self.x[2:4] = np.maximum(self.x[2:4], 1e-3)
        self.P = self._F @ self.P @ self._F.T + self._noise(0.05, 0.00625)

    def update(self, box):
        z = np.array(self._to_xywh(box))
        h = max(self.x[3], 1e-3)
        R = np.diag([0.05 * h] * 4) ** 2
        S = self._H @ self.P @ self._H.T + R
        K = self.P @ self._H.T @ np.linalg.inv(S)
        self.x = self.x + K @ (z - self._H @ self.x)
        self.P = (np.eye(8) - K @ self._H) @ self.P

    @property
    def box(self):
        cx, cy, w, h = self.x[:4]
        return (cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)

    @property
    def speed(self):
        return float(np.hypot(self.x[4], self.x[5]))


# ---------------------------------------------------------------- tracks
class Track:
    def __init__(self, tid, box, score):
        self.id = tid
        self.kf = KalmanBox(box)
        self.box = box
        self.last_box = box     # last *observed* box
        self.score = score
        self.hits = 1
        self.missed = 0
        self.confirmed = False
        self.side = 0          # last stable side of the counting line (-1 / +1)
        self.counted = False   # used by "unique" mode
        self.trail = []
        self.counted_at = None  # (monotonic time, direction) of the last count
        self.event_id = None    # DB id of the last count (for undo)

    def anchor(self, mode="center"):
        b = self.box
        if mode == "bottom":  # feet: best when the line is drawn on the floor
            return ((b[0] + b[2]) / 2, b[3])
        if mode == "top":     # head: best for steep/overhead-ish cameras with crowds
            return ((b[0] + b[2]) / 2, b[1] + (b[3] - b[1]) * 0.15)
        return ((b[0] + b[2]) / 2, (b[1] + b[3]) / 2)


def _direction_ok(t, box):
    """Detection must lie roughly where the track is heading (no stealing by people
    walking the other way). Tracks without a reliable velocity accept anything."""
    vx, vy = t.kf.x[4], t.kf.x[5]
    speed = (vx * vx + vy * vy) ** 0.5
    if t.hits < 4 or speed < 0.002:
        return True
    lb = t.last_box
    dx = (box[0] + box[2]) / 2 - (lb[0] + lb[2]) / 2
    dy = (box[1] + box[3]) / 2 - (lb[1] + lb[3]) / 2
    moved = (dx * dx + dy * dy) ** 0.5
    if moved < 0.25 * max(lb[3] - lb[1], 1e-3):
        return True  # small displacement: within noise, any direction is fine
    return (dx * vx + dy * vy) / (moved * speed) > 0.0


def _greedy_match(tracks, dets, min_score, allow_distance):
    """Return matches [(ti, di)], unmatched track idx, unmatched det idx."""
    pairs = []
    for ti, t in enumerate(tracks):
        pb = t.kf.box
        lost = t.missed > 0
        ph = max(pb[3] - pb[1], 1e-3)
        for di, (box, _) in enumerate(dets):
            hr = (box[3] - box[1]) / ph
            if not 0.7 < hr < 1.43:
                continue  # a person's height doesn't jump between frames: different person
            s = iou(pb, box)
            if lost and (s < max(min_score, 0.3) or (s < 0.5 and not _direction_ok(t, box))):
                continue  # re-attaching a hidden track needs strong, consistent evidence
            if s < min_score and allow_distance:
                if not _direction_ok(t, box):
                    continue
                # IoU fails for small/fast people: fall back to centre distance vs size
                pcx, pcy = (pb[0] + pb[2]) / 2, (pb[1] + pb[3]) / 2
                dcx, dcy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
                size = max(pb[3] - pb[1], box[3] - box[1], 1e-3)
                d = np.hypot(pcx - dcx, pcy - dcy) / size
                hr = (box[3] - box[1]) / max(pb[3] - pb[1], 1e-3)
                if d < 0.6 and 0.6 < hr < 1.6:
                    s = min_score * (1 - d / 0.6) * 0.99
                else:
                    continue
            elif s < min_score:
                continue
            pairs.append((s, ti, di))
    pairs.sort(reverse=True)
    used_t, used_d, matches = set(), set(), []
    for _, ti, di in pairs:
        if ti in used_t or di in used_d:
            continue
        used_t.add(ti)
        used_d.add(di)
        matches.append((ti, di))
    return (matches, [i for i in range(len(tracks)) if i not in used_t],
            [i for i in range(len(dets)) if i not in used_d])


class Tracker:
    def __init__(self, max_missed=30, min_hits=3):
        self.max_missed = max_missed    # frames a confirmed track survives unseen
        self.min_hits = min_hits
        self.tracks = []
        self._ids = itertools.count(1)
        self.frame = 0

    def update(self, dets, high_thr):
        """dets: [(box, score)] including low-confidence ones. Returns all tracks."""
        self.frame += 1
        for t in self.tracks:
            t.kf.predict()
        high = [d for d in dets if d[1] >= high_thr]
        low = [d for d in dets if d[1] < high_thr]

        # stage 1: all tracks vs confident detections
        m1, rest_t, rest_high = _greedy_match(self.tracks, high, 0.2, allow_distance=True)
        for ti, di in m1:
            self._hit(self.tracks[ti], *high[di])
        # stage 2: leftover confirmed tracks vs low-confidence detections (stricter IoU)
        rest = [self.tracks[i] for i in rest_t if self.tracks[i].confirmed]
        m2, _, _ = _greedy_match(rest, low, 0.35, allow_distance=False)
        matched2 = set()
        for ti, di in m2:
            self._hit(rest[ti], *low[di])
            matched2.add(id(rest[ti]))
        for i in rest_t:
            t = self.tracks[i]
            if id(t) not in matched2:
                t.missed += 1
                t.box = t.kf.box  # coast on the prediction while hidden
        # drop dead tracks: tentative ones die immediately, confirmed after max_missed,
        # and any unseen track whose prediction has left the frame (they walked out —
        # never let a newcomer at the same edge inherit their ID)
        def alive(t):
            if not t.confirmed:
                return t.missed == 0
            if t.missed == 0:
                return True
            cx, cy = (t.box[0] + t.box[2]) / 2, (t.box[1] + t.box[3]) / 2
            return t.missed <= self.max_missed and 0.0 < cx < 1.0 and 0.0 < cy < 1.0
        self.tracks = [t for t in self.tracks if alive(t)]
        for di in rest_high:
            self.tracks.append(Track(next(self._ids), *high[di]))
        for t in self.tracks:
            if t.missed == 0:
                t.trail.append(t.anchor())
                del t.trail[:-40]
        return self.tracks

    def _hit(self, t, box, score):
        t.kf.update(box)
        t.box = t.kf.box
        t.last_box = box
        t.score = score
        t.hits += 1
        t.missed = 0
        if t.hits >= self.min_hits:
            t.confirmed = True


# ---------------------------------------------------------------- counting
def side_of_line(p, a, b, margin):
    """-1 / +1 depending on which side of line a->b point p sits; 0 inside dead-zone.

    Points beyond the ends of the segment return 0 too, so people walking past the
    end of a short line are not counted.
    """
    ax, ay = a
    bx, by = b
    dx, dy = bx - ax, by - ay
    length2 = dx * dx + dy * dy or 1.0
    t = ((p[0] - ax) * dx + (p[1] - ay) * dy) / length2
    if t < -0.05 or t > 1.05:
        return 0
    dist = (dx * (p[1] - ay) - dy * (p[0] - ax)) / length2 ** 0.5
    if abs(dist) < margin:
        return 0
    return 1 if dist > 0 else -1


class Counter:
    """Turns tracks into count events.

    mode "line":   count when a confirmed track's anchor crosses the line (in / out).
    mode "unique": count every confirmed person once.
    """

    def __init__(self, on_event):
        self.on_event = on_event
        self.mode = "line"
        self.line = ((0.5, 0.0), (0.5, 1.0))
        self.margin = 0.02
        self.count_dirs = "both"  # both | in | out
        self.invert = False
        self.anchor = "center"
        self.undo_window = 3.0  # seconds: re-crossing back within this = never really passed
        self.frame = 0

    def configure(self, s):
        self.mode = s.get("mode", self.mode)
        line = s.get("line")
        if line and len(line) == 4:
            self.line = ((float(line[0]), float(line[1])), (float(line[2]), float(line[3])))
        self.count_dirs = s.get("count_dirs", self.count_dirs)
        self.invert = bool(s.get("invert", self.invert))
        self.anchor = s.get("anchor", self.anchor)
        self.margin = float(s.get("line_margin", self.margin))
        self.undo_window = float(s.get("undo_window", self.undo_window))

    def process(self, tracks, now=None):
        """now: timestamp in seconds (defaults to the monotonic clock)."""
        self.frame += 1
        now = time.monotonic() if now is None else now
        for t in tracks:
            if not t.confirmed:
                continue
            if self.mode == "unique":
                if not t.counted and t.missed == 0:
                    t.counted = True
                    self.on_event("pass", t)
                continue
            if t.missed:
                continue  # never count on a predicted (unseen) position
            s = side_of_line(t.anchor(self.anchor), *self.line, self.margin)
            if s == 0:
                continue
            if t.side == 0:
                t.side = s
                continue
            if s != t.side:
                direction = "in" if (s > 0) != self.invert else "out"
                t.side = s
                # stepping over the line and straight back = didn't pass: take the count back
                if t.counted_at and t.counted_at[1] != direction and now - t.counted_at[0] < self.undo_window:
                    t.counted_at = None
                    self.on_event("undo", t)
                    continue
                t.counted_at = (now, direction)
                if self.count_dirs in ("both", direction):
                    self.on_event(direction, t)
