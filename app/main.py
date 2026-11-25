"""Coral People Counter — web server entry point."""
import base64
import csv
import io
import logging
import os
import time
from datetime import date, datetime

from flask import Flask, Response, jsonify, render_template, request, send_from_directory

import camera
import cv2
import detector as det_mod
from db import Store
from pipeline import Pipeline

logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO"),
                    format="%(asctime)s %(levelname)s %(name)s: %(message)s")
if hasattr(time, "tzset"):
    time.tzset()

DATA_DIR = os.environ.get("DATA_DIR", "/data")
os.makedirs(DATA_DIR, exist_ok=True)
store = Store(os.path.join(DATA_DIR, "people.db"))
pipeline = Pipeline(store)
ADMIN_TOKEN = os.environ.get("ADMIN_TOKEN", "")

app = Flask(__name__)


def _admin_ok():
    return not ADMIN_TOKEN or request.headers.get("X-Admin-Token") == ADMIN_TOKEN


@app.route("/")
def index():
    return render_template("index.html", site_name=os.environ.get("SITE_NAME", "People Counter"),
                           admin_required=bool(ADMIN_TOKEN))


@app.route("/api/summary")
def summary():
    s = store.summary()
    s.update({
        "now_visible": pipeline.active,
        "fps": round(pipeline.fps, 1),
        "inference_ms": round(pipeline.inference_ms, 1),
        "backend": pipeline.backend,
        "status": pipeline.status,
        "source": pipeline.source_desc,
        "view": pipeline.settings.get("view", "angled"),
        "view_warning": pipeline.view_warning(),
        "server_time": datetime.now().isoformat(timespec="seconds"),
        "timezone": time.strftime("%Z"),
    })
    return jsonify(s)


@app.route("/api/hourly")
def hourly():
    d = request.args.get("date")
    day = date.fromisoformat(d) if d else date.today()
    return jsonify(store.hourly(day))


@app.route("/api/daily")
def daily():
    m = request.args.get("month")
    if m:
        y, mo = (int(v) for v in m.split("-"))
        return jsonify(store.daily(y, mo))
    return jsonify(store.last_days(int(request.args.get("days", 30))))


@app.route("/api/monthly")
def monthly():
    return jsonify(store.monthly(int(request.args.get("year", date.today().year))))


@app.route("/api/yearly")
def yearly():
    return jsonify(store.yearly())


@app.route("/api/profile")
def profile():
    return jsonify(store.hour_of_day_profile())


@app.route("/api/settings", methods=["GET", "POST"])
def settings():
    if request.method == "POST":
        if not _admin_ok():
            return jsonify({"error": "admin token required"}), 401
        return jsonify(pipeline.update_settings(request.get_json(force=True) or {}))
    return jsonify(pipeline.settings)


@app.route("/api/camera", methods=["GET", "POST"])
def camera_settings():
    if request.method == "POST":
        if not _admin_ok():
            return jsonify({"error": "admin token required"}), 401
        try:
            return jsonify(pipeline.set_camera(request.get_json(force=True) or {}))
        except ValueError as e:
            return jsonify({"error": str(e)}), 400
    return jsonify(pipeline.camera_config(masked=True))


@app.route("/api/camera/devices")
def camera_devices():
    return jsonify(camera.list_usb_devices())


@app.route("/api/camera/test", methods=["POST"])
def camera_test():
    if not _admin_ok():
        return jsonify({"error": "admin token required"}), 401
    cfg = pipeline.merge_camera(request.get_json(force=True) or {})
    if cfg.get("type") == "demo":
        return jsonify({"ok": True, "width": 960, "height": 540})
    if cfg.get("type") == "usb" and cfg.get("device") == pipeline.camera.get("device") \
            and pipeline.camera.get("type") == "usb":
        return jsonify({"ok": True, "note": "this webcam is already in use by the counter"})
    try:
        frame, w, h = camera.grab_test_frame(cfg)
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 200
    ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
    return jsonify({"ok": True, "width": w, "height": h,
                    "url": camera.mask_url(camera.stream_url(cfg)) if cfg.get("type") != "usb" else cfg.get("device"),
                    "jpeg": base64.b64encode(buf.tobytes()).decode()})


@app.route("/api/models")
def models():
    return jsonify({"models": det_mod.model_list(), "topdown": det_mod.topdown_status()})


@app.route("/api/recent")
def recent():
    rows = store.recent(min(200, int(request.args.get("limit", 40))))
    for r in rows:
        r["snapshot"] = os.path.exists(os.path.join(pipeline.snap_dir, f"{r['id']}.jpg"))
    return jsonify(rows)


@app.route("/snapshots/<int:event_id>.jpg")
def snapshot_file(event_id):
    return send_from_directory(pipeline.snap_dir, f"{event_id}.jpg")  # Flask 1.x compatible


@app.route("/api/events/<int:event_id>", methods=["DELETE"])
def delete_event(event_id):
    """Remove a wrong count (seen in the snapshot strip)."""
    if not _admin_ok():
        return jsonify({"error": "admin token required"}), 401
    store.delete_event(event_id)
    pipeline._remove_snapshot(event_id)
    return jsonify({"ok": True})


@app.route("/api/reset", methods=["POST"])
def reset():
    if not _admin_ok():
        return jsonify({"error": "admin token required"}), 401
    store.reset()
    for f in os.listdir(pipeline.snap_dir):
        if f.endswith(".jpg"):
            pipeline._remove_snapshot(f[:-4])
    return jsonify({"ok": True})


@app.route("/api/export.csv")
def export_csv():
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(["id", "local_time", "unix_ts", "direction"])
    w.writerows(store.export_rows())
    return Response(buf.getvalue(), mimetype="text/csv",
                    headers={"Content-Disposition": "attachment; filename=people-count.csv"})


@app.route("/video_feed")
def video_feed():
    return Response(pipeline.frames(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/snapshot.jpg")
def snapshot():
    jpg = pipeline.snapshot()
    if not jpg:
        return Response(status=503)
    return Response(jpg, mimetype="image/jpeg")


@app.route("/healthz")
def healthz():
    return jsonify({"status": pipeline.status, "backend": pipeline.backend})


def main():
    pipeline.start()
    port = int(os.environ.get("PORT", "8080"))
    try:
        from waitress import serve
        serve(app, host="0.0.0.0", port=port, threads=16)
    except ImportError:
        app.run(host="0.0.0.0", port=port, threaded=True)


if __name__ == "__main__":
    main()
