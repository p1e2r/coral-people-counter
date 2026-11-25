"""SQLite storage for crossing events and settings.

All aggregation is done in the container's local time zone (set TZ).
"""
import calendar
import json
import sqlite3
import threading
import time
from datetime import date, datetime

_LOCAL = "datetime(ts, 'unixepoch', 'localtime')"


class Store:
    def __init__(self, path):
        self.path = path
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(path, check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ts REAL NOT NULL,
                direction TEXT NOT NULL DEFAULT 'pass'
            );
            CREATE INDEX IF NOT EXISTS idx_events_ts ON events(ts);
            CREATE TABLE IF NOT EXISTS settings (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );
            """
        )
        self._conn.commit()

    # ---------- writes ----------
    def add_event(self, direction="pass", ts=None):
        with self._lock:
            cur = self._conn.execute(
                "INSERT INTO events(ts, direction) VALUES (?, ?)",
                (ts if ts is not None else time.time(), direction),
            )
            self._conn.commit()
            return cur.lastrowid

    def delete_event(self, event_id):
        with self._lock:
            self._conn.execute("DELETE FROM events WHERE id = ?", (event_id,))
            self._conn.commit()

    def recent(self, limit=40):
        return [{"id": i, "ts": ts, "direction": d} for i, ts, d in self._q(
            "SELECT id, ts, direction FROM events ORDER BY id DESC LIMIT ?", (limit,))]

    def add_many(self, rows):
        with self._lock:
            self._conn.executemany("INSERT INTO events(ts, direction) VALUES (?, ?)", rows)
            self._conn.commit()

    def reset(self):
        with self._lock:
            self._conn.execute("DELETE FROM events")
            self._conn.commit()

    # ---------- settings ----------
    def get_settings(self):
        with self._lock:
            rows = self._conn.execute("SELECT key, value FROM settings").fetchall()
        return {k: json.loads(v) for k, v in rows}

    def save_settings(self, values):
        with self._lock:
            self._conn.executemany(
                "INSERT INTO settings(key, value) VALUES (?, ?) "
                "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                [(k, json.dumps(v)) for k, v in values.items()],
            )
            self._conn.commit()

    # ---------- reads ----------
    def _q(self, sql, args=()):
        with self._lock:
            return self._conn.execute(sql, args).fetchall()

    def _count_since(self, start_ts):
        return self._q("SELECT COUNT(*) FROM events WHERE ts >= ?", (start_ts,))[0][0]

    def summary(self):
        now = datetime.now()
        hour = now.replace(minute=0, second=0, microsecond=0).timestamp()
        day = now.replace(hour=0, minute=0, second=0, microsecond=0).timestamp()
        month = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0).timestamp()
        year = now.replace(month=1, day=1, hour=0, minute=0, second=0, microsecond=0).timestamp()
        total, first = self._q("SELECT COUNT(*), MIN(ts) FROM events")[0]
        dirs = dict(self._q(
            "SELECT direction, COUNT(*) FROM events WHERE ts >= ? GROUP BY direction", (day,)))
        busiest = self._q(
            f"SELECT strftime('%Y-%m-%d', {_LOCAL}) d, COUNT(*) c FROM events "
            "GROUP BY d ORDER BY c DESC LIMIT 1")
        return {
            "this_hour": self._count_since(hour),
            "today": self._count_since(day),
            "this_month": self._count_since(month),
            "this_year": self._count_since(year),
            "all_time": total,
            "today_in": dirs.get("in", 0),
            "today_out": dirs.get("out", 0),
            "first_event": first,
            "busiest_day": {"date": busiest[0][0], "count": busiest[0][1]} if busiest else None,
        }

    def hourly(self, day: date):
        start = datetime(day.year, day.month, day.day).timestamp()
        end = start + 86400 + 3600  # covers DST-long days; filtered by date below
        rows = self._q(
            f"SELECT CAST(strftime('%H', {_LOCAL}) AS INTEGER) h, COUNT(*) FROM events "
            f"WHERE ts >= ? AND ts < ? AND strftime('%Y-%m-%d', {_LOCAL}) = ? GROUP BY h",
            (start - 3600, end, day.isoformat()),
        )
        counts = dict(rows)
        return [{"label": f"{h:02d}:00", "count": counts.get(h, 0)} for h in range(24)]

    def daily(self, year: int, month: int):
        days = calendar.monthrange(year, month)[1]
        start = datetime(year, month, 1).timestamp()
        end = datetime(year + (month == 12), month % 12 + 1, 1).timestamp()
        rows = self._q(
            f"SELECT CAST(strftime('%d', {_LOCAL}) AS INTEGER) d, COUNT(*) FROM events "
            "WHERE ts >= ? AND ts < ? GROUP BY d",
            (start, end),
        )
        counts = dict(rows)
        return [{"label": f"{year}-{month:02d}-{d:02d}", "count": counts.get(d, 0)}
                for d in range(1, days + 1)]

    def last_days(self, n=30):
        today = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        start = today.timestamp() - (n - 1) * 86400 - 3600
        rows = dict(self._q(
            f"SELECT strftime('%Y-%m-%d', {_LOCAL}) d, COUNT(*) FROM events "
            "WHERE ts >= ? GROUP BY d", (start,)))
        out = []
        for i in range(n - 1, -1, -1):
            d = date.fromordinal(today.date().toordinal() - i).isoformat()
            out.append({"label": d, "count": rows.get(d, 0)})
        return out

    def monthly(self, year: int):
        start = datetime(year, 1, 1).timestamp()
        end = datetime(year + 1, 1, 1).timestamp()
        rows = dict(self._q(
            f"SELECT CAST(strftime('%m', {_LOCAL}) AS INTEGER) m, COUNT(*) FROM events "
            "WHERE ts >= ? AND ts < ? GROUP BY m", (start, end)))
        return [{"label": calendar.month_abbr[m], "count": rows.get(m, 0)} for m in range(1, 13)]

    def yearly(self):
        rows = self._q(
            f"SELECT strftime('%Y', {_LOCAL}) y, COUNT(*) FROM events GROUP BY y ORDER BY y")
        if not rows:
            return [{"label": str(datetime.now().year), "count": 0}]
        return [{"label": y, "count": c} for y, c in rows]

    def hour_of_day_profile(self):
        """Average-free 'busiest hours' profile across all time."""
        rows = dict(self._q(
            f"SELECT CAST(strftime('%H', {_LOCAL}) AS INTEGER) h, COUNT(*) FROM events GROUP BY h"))
        return [{"label": f"{h:02d}:00", "count": rows.get(h, 0)} for h in range(24)]

    def export_rows(self):
        return self._q(
            f"SELECT id, {_LOCAL}, ts, direction FROM events ORDER BY ts")
