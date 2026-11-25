import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
APP = os.environ.get("APP_DIR") or os.path.join(HERE, "..", "..", "app")
if not os.path.isdir(APP):
    APP = "/app"  # running inside the Docker image
sys.path.insert(0, APP)
DATA = os.environ.get("ACC_DATA", os.path.join(HERE, "data"))
