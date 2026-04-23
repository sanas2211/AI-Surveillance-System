from pathlib import Path
import sys

FILE = Path(__file__).resolve()
ROOT = FILE.parent

if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

# Sources
SOURCES_LIST = ["Image", "Video", "Webcam"]

# Model paths
DETECTION_MODEL_DIR = ROOT / "weights" / "detection"

DETECTION_MODEL_LIST = [
    "yolov8n.pt",
    "yolov8s.pt",
    "yolov8m.pt",
    "yolov8l.pt",
    "yolov8x.pt"
]

# Counters
OBJECT_COUNTER = None
OBJECT_COUNTER1 = None