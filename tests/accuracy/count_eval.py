"""End-to-end counting accuracy on real pedestrian video.

Video: OpenCV's samples/data/vtest.avi (768x576, 10 fps, 795 frames, angled view of a plaza).
Ground truth: 31 people cross a vertical line at x = 45 %, counted by hand from a slit-scan
of the line and checked frame by frame (frames 334-342, 640-696 and 776-784 verified).

Runs the app's real detector + tracker + counter. Usage:
    python tests/accuracy/count_eval.py [--models a,b] [--check]
--check exits non-zero if the default model's counting F1 is below 95 %.
"""
import argparse
import os

import cv2

from common import DATA
import detector as D
import tracker as T

GT = [11, 64, 97, 163, 184, 224, 260, 284, 296, 306, 325, 338, 342, 369, 456, 496, 535, 571, 588,
      606, 648, 650, 671, 692, 694, 713, 737, 756, 777, 779, 783]
LINE = [0.45, 0.10, 0.45, 1.0]
FPS = 10
TOLERANCE = 12  # frames


def load_frames():
    cap = cv2.VideoCapture(os.path.join(DATA, "vtest.avi"))
    frames = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
    if not frames:
        raise SystemExit("could not read vtest.avi (run tests/accuracy/fetch_data.sh)")
    return frames


def run(frames, key, thr=0.45, tiles="auto"):
    det = D.TFLiteDetector(key)
    tr = T.Tracker(max_missed=int(FPS * 1.5), min_hits=3)
    events = []
    ct = T.Counter(lambda d, t: events.append((d, fi)))
    ct.configure({"line": LINE, "anchor": "center"})
    for fi, rgb in enumerate(frames):
        ct.process(tr.update(det.detect(rgb, max(0.15, thr * 0.5), tiles=tiles), thr), now=fi / FPS)
    counts = [f for d, f in events if d in ("in", "out")]
    for u in [f for d, f in events if d == "undo"]:
        prev = [c for c in counts if c <= u]
        if prev:
            counts.remove(prev[-1])
    gt, tp = sorted(GT), 0
    for c in sorted(counts):  # earliest-first matching is optimal for 1-D intervals
        m = [g for g in gt if abs(g - c) <= TOLERANCE]
        if m:
            gt.remove(min(m))
            tp += 1
    fp, fn = len(counts) - tp, len(gt)
    return {"model": key, "backend": det.backend, "counted": len(counts), "tp": tp, "fp": fp, "fn": fn,
            "f1": 2 * tp / max(2 * tp + fp + fn, 1)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="ssd_mobilenet_v2,mobiledet,efficientdet_lite0,efficientdet_lite1,"
                                        "efficientdet_lite2,efficientdet_lite3x")
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    frames = load_frames()
    print(f"ground truth: {len(GT)} crossings in {len(frames)} frames\n")
    print(f"{'model':22s} {'counted':>7s} {'error':>7s} {'correct':>7s} {'extra':>5s} {'missed':>6s} {'F1':>6s}")
    results = {}
    for key in a.models.split(","):
        if not D.model_available(key):
            print(f"{key:22s} (not installed)")
            continue
        r = run(frames, key)
        results[key] = r
        err = 100 * (r["counted"] - len(GT)) / len(GT)
        star = " *default" if key == D.DEFAULT_MODEL else ""
        print(f"{key:22s} {r['counted']:7d} {err:+6.1f}% {r['tp']:7d} {r['fp']:5d} {r['fn']:6d} {r['f1']*100:5.1f}%{star}",
              flush=True)
    if a.check:
        r = results.get(D.DEFAULT_MODEL) or run(frames, D.DEFAULT_MODEL)
        print(f"\ndefault model {D.DEFAULT_MODEL}: F1 {r['f1']*100:.1f}% (gate: 95%)")
        raise SystemExit(0 if r["f1"] >= 0.95 else 1)


if __name__ == "__main__":
    main()
