"""Fill the database with ~2 years of fake history so the charts have something to show.

Usage (inside the container):
    docker compose exec people-counter python3 /scripts/seed_demo_data.py
or locally:
    DATA_DIR=./data python3 scripts/seed_demo_data.py
"""
import math
import os
import random
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "app"))
from db import Store  # noqa: E402

data_dir = os.environ.get("DATA_DIR", "/data")
os.makedirs(data_dir, exist_ok=True)
store = Store(os.path.join(data_dir, "people.db"))

now = time.time()
start = now - 730 * 86400
rows, t = [], start
while t < now:
    lt = time.localtime(t)
    hour_curve = max(0.0, math.sin((lt.tm_hour - 6) / 16 * math.pi))  # busy 06:00-22:00
    weekend = 1.4 if lt.tm_wday >= 5 else 1.0
    rate = 40 * hour_curve * weekend * (0.7 + 0.3 * (t - start) / (now - start))  # people/hour
    n = random.randint(int(rate * 0.7), int(rate * 1.3) + 1) if rate else 0
    for _ in range(n):
        rows.append((t + random.random() * 3600, random.choice(("in", "out"))))
    t += 3600
store.add_many(rows)
print(f"inserted {len(rows):,} events")
