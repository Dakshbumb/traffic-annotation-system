"""
Autolabel Worker with DeepSORT Tracking (Production-Ready)
- YOLO detection with error handling and retries
- DeepSORT tracking with failure recovery
- Performance monitoring and logging
- Annotation retry queue for failed saves
"""

import os
import time
import threading
import logging
from datetime import datetime
from typing import Optional

# GPU mode enabled - will use CUDA if available
# os.environ["CUDA_VISIBLE_DEVICES"] = ""  # Commented out to enable GPU
# os.environ["TORCH_CUDA_AVAILABLE"] = "0"

import cv2
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort

from config import (
    UPLOAD_DIR,
    YOLO_MODEL_NAME,
    YOLO_COCO_CLASS_MAP,
    ALLOWED_CLASSES,
    AUTO_DEFAULT_CONF_TH,
    AUTO_DEFAULT_NMS_IOU,
    DEEPSORT_MAX_AGE,
    DEEPSORT_N_INIT,
    DEEPSORT_MAX_COSINE_DIST,
    DEEPSORT_NN_BUDGET,
    ENABLE_CLAHE,
    DEVICE,
)
from preprocessing import preprocess_frame
from postprocessing import apply_all_filters, get_class_confidence_threshold
from monitoring import PerformanceMonitor, get_memory_usage, logger as monitor_logger
from retry_queue import RetryQueue, with_retry
from database import SessionLocal
import crud
import models
from job_store import job_store, update_job

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("autolabel")

# -----------------------------
# Model Loading with Retry
# -----------------------------
MODEL_LOAD_MAX_RETRIES = 3
MODEL_LOAD_BASE_DELAY = 2.0

yolo: Optional[YOLO] = None


def load_yolo_model() -> YOLO:
    """Load YOLO model with retry and exponential backoff."""
    global yolo
    
    for attempt in range(1, MODEL_LOAD_MAX_RETRIES + 1):
        try:
            logger.info(f"Loading YOLO model (attempt {attempt}/{MODEL_LOAD_MAX_RETRIES})...")
            model = YOLO(YOLO_MODEL_NAME)
            model.to(DEVICE)
            logger.info(f"YOLO model loaded: {YOLO_MODEL_NAME} on {DEVICE}")
            return model
        except Exception as e:
            logger.error(f"Failed to load YOLO model: {e}")
            if attempt < MODEL_LOAD_MAX_RETRIES:
                delay = MODEL_LOAD_BASE_DELAY * (2 ** (attempt - 1))
                logger.info(f"Retrying in {delay}s...")
                time.sleep(delay)
            else:
                logger.critical("Failed to load YOLO model after all retries")
                raise RuntimeError(f"Could not load YOLO model: {e}")
    
    return None


# Load model at startup
try:
    yolo = load_yolo_model()
except Exception as e:
    logger.critical(f"YOLO model initialization failed: {e}")


# -----------------------------
# Retry Queue for Annotations
# -----------------------------
retry_queue = RetryQueue(max_attempts=3, base_delay=1.0)


def save_annotations_handler(data: dict):
    """Handler for saving annotations from retry queue."""
    db = SessionLocal()
    try:
        annotations = data["annotations"]
        db.bulk_save_objects(annotations)
        db.commit()
        logger.info(f"Retry: Saved {len(annotations)} annotations")
    finally:
        db.close()


retry_queue.register_handler("save_annotations", save_annotations_handler)
retry_queue.start()


# -----------------------------
# Tracker Factory
# -----------------------------
def create_tracker() -> DeepSort:
    """Create a new DeepSORT tracker instance."""
    return DeepSort(
        max_age=DEEPSORT_MAX_AGE,
        n_init=DEEPSORT_N_INIT,
        nms_max_overlap=1.0,
        max_cosine_distance=DEEPSORT_MAX_COSINE_DIST,
        nn_budget=DEEPSORT_NN_BUDGET,
        embedder_gpu=DEVICE == "cuda",  # Use GPU if available
    )


# -----------------------------
# Main Job Runner
# -----------------------------
def _run_job(job_id: str):
    """
    Run the auto-labeling job with full error handling.
    """
    global yolo
    
    job = job_store.get(job_id)
    if not job:
        return

    params = job["params"]
    video_id = params["video_id"]
    conf_threshold = params.get("confidence_threshold", AUTO_DEFAULT_CONF_TH)
    frame_stride = params.get("frame_stride", 1)
    overwrite = params.get("overwrite", True)

    db = SessionLocal()
    monitor = PerformanceMonitor(log_interval=100)
    tracker = None
    cap = None

    try:
        # Ensure model is loaded
        if yolo is None:
            yolo = load_yolo_model()
        
        # Update job status
        update_job(job_id, {
            "status": "running",
            "progress": 0.0,
            "updated_at": datetime.utcnow().isoformat(),
        })

        # Get video
        video = crud.get_video(db, video_id)
        if not video:
            raise RuntimeError(f"Video not found: {video_id}")

        video_path = os.path.join(UPLOAD_DIR, video.filename)
        if not os.path.exists(video_path):
            raise RuntimeError(f"Video file not found: {video_path}")

        # Open video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open video: {video_path}")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        logger.info(f"Processing video {video_id}: {total_frames} frames @ {fps:.1f} fps")

        # Delete existing annotations if overwrite
        if overwrite:
            deleted = db.query(models.Annotation).filter(
                models.Annotation.video_id == video_id
            ).delete()
            db.commit()
            logger.info(f"Deleted {deleted} existing annotations")

        # Initialize tracker
        tracker = create_tracker()
        consecutive_empty_tracks = 0
        MAX_EMPTY_BEFORE_RESET = 30

        # Start monitoring
        monitor.start()

        frame_idx = 0
        annotations_batch = []
        batch_size = 100

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % frame_stride != 0:
                frame_idx += 1
                continue

            monitor.frame_start()
            orig_h, orig_w = frame.shape[:2]

            # --- PREPROCESSING ---
            try:
                processed_frame, scale = preprocess_frame(frame, apply_enhancement=ENABLE_CLAHE)
            except Exception as e:
                logger.warning(f"Preprocessing failed at frame {frame_idx}: {e}")
                processed_frame, scale = frame, 1.0

            # --- DETECTION ---
            detections = []
            try:
                results = yolo(
                    processed_frame,
                    verbose=False,
                    conf=conf_threshold,
                    iou=AUTO_DEFAULT_NMS_IOU,
                )[0]

                for box in results.boxes:
                    cls_idx = int(box.cls[0])
                    class_name = YOLO_COCO_CLASS_MAP.get(cls_idx)
                    if class_name is None or class_name not in ALLOWED_CLASSES:
                        continue

                    conf = float(box.conf[0])
                    class_conf_th = get_class_confidence_threshold(class_name)
                    if conf < class_conf_th:
                        continue

                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    if scale < 1.0:
                        x1, y1, x2, y2 = x1/scale, y1/scale, x2/scale, y2/scale

                    if not apply_all_filters(x1, y1, x2, y2, orig_w, orig_h, conf, class_name):
                        continue

                    bbox = [x1, y1, x2 - x1, y2 - y1]
                    detections.append((bbox, conf, class_name))

                monitor.record_detections(len(detections))

            except Exception as e:
                logger.error(f"Detection failed at frame {frame_idx}: {e}")
                monitor.record_detection_failure()
                # Continue with empty detections
                detections = []

            # --- TRACKING ---
            tracks = []
            try:
                tracks = tracker.update_tracks(detections, frame=frame)
                
                # Check for all tracks lost scenario
                confirmed_tracks = [t for t in tracks if t.is_confirmed()]
                if len(confirmed_tracks) == 0 and len(detections) > 0:
                    consecutive_empty_tracks += 1
                else:
                    consecutive_empty_tracks = 0
                
                # Reset tracker if all tracks lost for too long
                if consecutive_empty_tracks >= MAX_EMPTY_BEFORE_RESET:
                    logger.warning(f"Resetting tracker at frame {frame_idx} (no tracks for {MAX_EMPTY_BEFORE_RESET} frames)")
                    tracker = create_tracker()
                    consecutive_empty_tracks = 0
                    monitor.record_tracker_reset()

            except Exception as e:
                logger.error(f"Tracking failed at frame {frame_idx}: {e}")
                monitor.record_tracking_failure()
                
                # Try to reset tracker
                try:
                    tracker = create_tracker()
                    monitor.record_tracker_reset()
                except Exception as e2:
                    logger.critical(f"Failed to reset tracker: {e2}")

            # --- ANNOTATIONS ---
            track_ids = set()
            for track in tracks:
                if not track.is_confirmed():
                    continue

                track_id = track.track_id
                track_ids.add(track_id)
                ltrb = track.to_ltrb()
                class_name = track.det_class if track.det_class else "object"

                annotation = models.Annotation(
                    video_id=video_id,
                    frame_index=frame_idx,
                    track_id=track_id,
                    class_label=class_name,
                    x1=float(ltrb[0]),
                    y1=float(ltrb[1]),
                    x2=float(ltrb[2]),
                    y2=float(ltrb[3]),
                    extra_meta={"confidence": float(track.det_conf) if track.det_conf else 0.0}
                )
                annotations_batch.append(annotation)

            # Batch commit with retry
            if len(annotations_batch) >= batch_size:
                try:
                    db.bulk_save_objects(annotations_batch)
                    db.commit()
                except Exception as e:
                    logger.error(f"Failed to save batch: {e}")
                    db.rollback()
                    # Queue for retry
                    retry_queue.enqueue({"annotations": annotations_batch}, "save_annotations")
                
                annotations_batch = []

            # Update progress and monitoring
            monitor.frame_end(frame_idx, track_ids)
            
            if frame_idx % 10 == 0:
                progress = frame_idx / total_frames
                update_job(job_id, {
                    "progress": progress,
                    "updated_at": datetime.utcnow().isoformat(),
                })

            frame_idx += 1

        # Commit remaining
        if annotations_batch:
            try:
                db.bulk_save_objects(annotations_batch)
                db.commit()
            except Exception as e:
                logger.error(f"Failed to save final batch: {e}")
                retry_queue.enqueue({"annotations": annotations_batch}, "save_annotations")

        # Get stats
        summary = monitor.stop()
        class_counts = {}
        ann_stats = db.query(models.Annotation.class_label).filter(
            models.Annotation.video_id == video_id
        ).all()
        for (label,) in ann_stats:
            class_counts[label] = class_counts.get(label, 0) + 1

        # Complete
        update_job(job_id, {
            "status": "completed",
            "progress": 1.0,
            "updated_at": datetime.utcnow().isoformat(),
        })

        logger.info(f"Job {job_id} completed!")
        logger.info(f"Performance: {summary}")
        logger.info(f"Annotations by class: {class_counts}")

    except Exception as e:
        update_job(job_id, {
            "status": "failed",
            "error_message": str(e),
            "updated_at": datetime.utcnow().isoformat(),
        })
        logger.error(f"Job {job_id} FAILED: {e}")
        import traceback
        traceback.print_exc()

    finally:
        if cap:
            cap.release()
        db.close()


def _worker_loop():
    """Background worker that processes queued jobs."""
    logger.info("Worker loop started")
    while True:
        for job in list(job_store.values()):
            if job["status"] == "queued":
                _run_job(job["job_id"])
        time.sleep(2)


def start_worker():
    """Start the background worker thread."""
    thread = threading.Thread(target=_worker_loop, daemon=True)
    thread.start()
    logger.info("Worker thread running")
