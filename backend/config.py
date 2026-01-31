from dotenv import load_dotenv
from pathlib import Path
import os

# ---------- Load .env ----------
load_dotenv()

# ---------- Base paths ----------

# Folder where backend code lives (where main.py is)
BASE_DIR = Path(__file__).resolve().parent

# Database
DB_FILE = BASE_DIR / "traffic_annotations.db"
DATABASE_URL = os.getenv("DATABASE_URL", f"sqlite:///{DB_FILE}")

# Single shared uploads folder  -> backend/uploads
UPLOAD_DIR = BASE_DIR / "uploads"



# ---------- PERFORMANCE / ML CONFIG ----------

# YOLO model to use (yolov8m = medium, better accuracy for dashcam footage)
YOLO_MODEL_NAME = os.getenv("YOLO_MODEL_NAME", "yolov8m.pt")

# ---------- Compute device (Auto-detect GPU with compatibility check) ----------
import torch
import warnings

def get_device():
    """Detect available device with compatibility check."""
    if not torch.cuda.is_available():
        return "cpu"
    
    try:
        # Test if CUDA actually works by running a small operation
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            test_tensor = torch.zeros(1).cuda()
            _ = test_tensor + 1
        return "cuda"
    except Exception as e:
        print(f"[config] CUDA test failed, falling back to CPU: {e}")
        return "cpu"

DEVICE = get_device()
USE_FP16 = DEVICE == "cuda"  # Use half precision on GPU for speedup
print(f"[config] Using device: {DEVICE}")


# Exports folder -> backend/exports
EXPORT_DIR = BASE_DIR / "exports"

# (Optional) models folder -> backend/models  (you can use later if you want)
MODELS_DIR = BASE_DIR / "models"

# Create folders if they don't exist
for p in (UPLOAD_DIR, EXPORT_DIR, MODELS_DIR):
    p.mkdir(parents=True, exist_ok=True)

# ---------- CORS settings ----------

CORS_ORIGINS = os.getenv("CORS_ORIGINS", "*")

if CORS_ORIGINS == "*":
    CORS_ORIGIN_LIST = ["*"]
else:
    CORS_ORIGIN_LIST = [o.strip() for o in CORS_ORIGINS.split(",") if o.strip()]


# ---------- Autolabel / image size settings ----------

# Max frame width for autolabel processing (in pixels).
# If a video frame is wider than this, we downscale before YOLO/DeepSORT.
MAX_FRAME_WIDTH = int(os.getenv("MAX_FRAME_WIDTH", "1280"))


# ---------- Autolabel / ML settings ----------

# YOLO model name or path (yolov8m = medium, better accuracy for dashcam footage)
YOLO_MODEL_NAME = os.getenv("YOLO_MODEL_NAME", "yolov8m.pt")

# How many jobs can be queued / run in parallel (used later if we add limits)
AUTOLABEL_MAX_JOBS = int(os.getenv("AUTOLABEL_MAX_JOBS", "8"))
AUTOLABEL_MAX_WORKERS = int(os.getenv("AUTOLABEL_MAX_WORKERS", "1"))

# ------------------------------
# Autolabel / detection settings
# ------------------------------

# Only keep these logical classes in our system
ALLOWED_CLASSES = [
    "person",
    "car",
    "truck",
    "bus",
    "motorcycle",
    "bicycle",
]

# Mapping from COCO / YOLOv8 class indices to names we care about
# (indices are COCO ids used by yolov8n.pt)
YOLO_COCO_CLASS_MAP = {
    0: "person",
    1: "bicycle",
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck",
    # we ignore all others
}

# Default thresholds for autolabel jobs
AUTO_DEFAULT_CONF_TH = float(os.getenv("AUTO_DEFAULT_CONF_TH", "0.35"))  # Higher to reduce false positives
AUTO_DEFAULT_NMS_IOU = float(os.getenv("AUTO_DEFAULT_NMS_IOU", "0.45"))  # Standard NMS
AUTO_DEFAULT_FRAME_STRIDE = 1

# ------------------------------
# Per-class confidence thresholds (optional tuning)
# ------------------------------
CLASS_CONF_THRESHOLDS = {
    "car": 0.35,         # Higher to reduce false positives on shadows
    "truck": 0.35,       # Same for trucks
    "bus": 0.35,         # Same for buses
    "motorcycle": 0.40,  # Higher for smaller objects
    "bicycle": 0.40,     # Same for bicycles  
    "person": 0.50,      # Much higher to reduce false positives
}

# ------------------------------
# Post-processing filters
# ------------------------------
MIN_BOX_AREA = int(os.getenv("MIN_BOX_AREA", "800"))           # Higher minimum to filter shadows
ASPECT_RATIO_MIN = float(os.getenv("ASPECT_RATIO_MIN", "0.25")) # Reasonable ratio bounds
ASPECT_RATIO_MAX = float(os.getenv("ASPECT_RATIO_MAX", "3.5"))
EDGE_MARGIN = int(os.getenv("EDGE_MARGIN", "10"))              # Pixels from edge to filter

# ------------------------------
# DeepSORT tracking parameters
# ------------------------------
DEEPSORT_MAX_AGE = int(os.getenv("DEEPSORT_MAX_AGE", "70"))           # Frames to keep track alive
DEEPSORT_N_INIT = int(os.getenv("DEEPSORT_N_INIT", "3"))              # Frames before confirmation
DEEPSORT_MAX_COSINE_DIST = float(os.getenv("DEEPSORT_MAX_COSINE_DIST", "0.3"))
DEEPSORT_NN_BUDGET = int(os.getenv("DEEPSORT_NN_BUDGET", "100"))      # Max samples per track

# ------------------------------
# Preprocessing options
# ------------------------------
ENABLE_CLAHE = os.getenv("ENABLE_CLAHE", "true").lower() == "true"    # Low-light enhancement
CLAHE_CLIP_LIMIT = float(os.getenv("CLAHE_CLIP_LIMIT", "2.0"))
CLAHE_TILE_SIZE = int(os.getenv("CLAHE_TILE_SIZE", "8"))
YOLO_INPUT_SIZE = int(os.getenv("YOLO_INPUT_SIZE", "1280"))           # 1280 for better small object detection
