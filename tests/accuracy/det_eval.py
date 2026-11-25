"""Person-detection accuracy on COCO128 (128 labelled photos, 254 people).

Uses the app's real detect() (shape filter, duplicate suppression, tiling). Reports AP50 and,
for each model, the best F1 and its threshold — over all people and over people at least
60 px tall (the size that matters for counting).
"""
import argparse
import glob
import os

import cv2
import numpy as np

from common import DATA
import detector as D


def load():
    data = []
    for p in sorted(glob.glob(os.path.join(DATA, "coco128", "images", "train2017", "*.jpg"))):
        im = cv2.imread(p)
        h, w = im.shape[:2]
        lab = p.replace(os.sep + "images" + os.sep, os.sep + "labels" + os.sep).rsplit(".", 1)[0] + ".txt"
        gt = []
        if os.path.exists(lab):
            for line in open(lab):
                c, cx, cy, bw, bh = line.split()[:5]
                if int(c) == 0:
                    cx, cy, bw, bh = map(float, (cx, cy, bw, bh))
                    gt.append((cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2, bh * h))
        data.append((cv2.cvtColor(im, cv2.COLOR_BGR2RGB), gt, h))
    if not data:
        raise SystemExit("COCO128 not found (run tests/accuracy/fetch_data.sh)")
    return data


def evaluate(data, key, tiles="1"):
    det = D.TFLiteDetector(key)
    rows = []
    for rgb, gt, H in data:
        preds = sorted(det.detect(rgb, 0.1, tiles=tiles), key=lambda d: -d[1])
        used = set()
        for box, s in preds:
            best, bi = 0, -1
            for i, g in enumerate(gt):
                if i not in used:
                    v = D.iou(box, g[:4])
                    if v > best:
                        best, bi = v, i
            ph = (box[3] - box[1]) * H
            if best >= 0.5:
                used.add(bi)
                rows.append((s, 1, gt[bi][4] >= 60, ph >= 60))
            else:
                rows.append((s, 0, False, ph >= 60))
    n_all = sum(len(g) for _, g, _ in data)
    n_big = sum(1 for _, g, _ in data for x in g if x[4] >= 60)
    rows.sort(key=lambda r: -r[0])
    tp = np.cumsum([r[1] for r in rows])
    fp = np.cumsum([1 - r[1] for r in rows])
    rec, prec = tp / n_all, tp / np.maximum(tp + fp, 1)
    mprec = np.maximum.accumulate(prec[::-1])[::-1]
    ap50 = float(np.sum(np.diff(np.concatenate([[0], rec])) * mprec))
    out = {"ap50": ap50}
    for scope, n in (("all", n_all), ("big", n_big)):
        best = (0, 0, 0, 0)
        for thr in np.arange(0.15, 0.81, 0.05):
            if scope == "all":
                sel = [r for r in rows if r[0] >= thr]
            else:
                sel = [r for r in rows if r[0] >= thr and (r[2] or (not r[1] and r[3]))]
            t = sum(r[1] for r in sel)
            p, rc = t / max(len(sel), 1), t / n
            f1 = 2 * p * rc / max(p + rc, 1e-9)
            if f1 > best[0]:
                best = (f1, thr, p, rc)
        out[scope] = best
    return out, n_all, n_big


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="ssd_mobilenet_v2,mobiledet,efficientdet_lite0,efficientdet_lite1,"
                                        "efficientdet_lite2,efficientdet_lite3,efficientdet_lite3x")
    a = ap.parse_args()
    data = load()
    print(f"{'model':22s} {'AP50':>5s} | best F1 all people (thr, P, R) | best F1 people >=60px (thr, P, R)")
    for key in a.models.split(","):
        if not D.model_available(key):
            continue
        r, n_all, n_big = evaluate(data, key)
        f = lambda b: f"{b[0]*100:4.1f} @{b[1]:.2f} (P {b[2]*100:4.1f}, R {b[3]*100:4.1f})"
        print(f"{key:22s} {r['ap50']*100:5.1f} | {f(r['all'])} | {f(r['big'])}", flush=True)
    print(f"\nground truth: {n_all} people in {len(data)} images, {n_big} at least 60 px tall")


if __name__ == "__main__":
    main()
