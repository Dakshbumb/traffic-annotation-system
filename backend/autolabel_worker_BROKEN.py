"""
Autolabel Worker
YOLOv8 + DeepSORT
CPU-only, stable, production-safe
"""

import os
import time
import threading
from datetime import datetime
from typing import List
def box_center(x1, y1, x2, y2):
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


# ===============================
# HARD DISABLE CUDA (RTX 50xx FIX)
# ===============================
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["TORCH_CUDA_AVAILABLE"] = "0"

import torch
torch.set_grad_enabled(False)

import cv2
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort

from config import UPLOAD_DIR, YOLO_MODEL_NAME, MAX_FRAME_WIDTH
from database import SessionLocal
import crud
import schemas
import models
from job_store import job_store, update_job

# ===============================
# CLASS MAP (traffic-specific)
# ===============================
CLASS_MAP = {
    0: "person",
    1: "bicycle",
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck",
}

# ===============================
# LOAD MODELS (CPU ONLY)
# ===============================
print(f"[autolabel] Loading YOLO model '{YOLO_MODEL_NAME}' on CPU...")
yolo_model = YOLO(YOLO_MODEL_NAME)
yolo_model.to("cpu")
print("[autolabel] YOLO loaded successfully (CPU-only).")

tracker = DeepSort(
    max_age=30,
    n_init=2,
    max_iou_distance=0.7,
    embedder="mobilenet",
    embedder_gpu=False,
    half=False,
)

# ===============================
# JOB EXECUTION
# ===============================
def _run_single_job(job_id: str) -> None:
    job = job_store.get(job_id)
    if not job:
        return

    params = job["params"]
    video_id = params["video_id"]
    frame_stride = params.get("frame_stride", 1)
    start_frame = params.get("start_frame", 0)
    end_frame = params.get("end_frame")
    conf_th = params.get("confidence_threshold", 0.4)
    overwrite = params.get("overwrite", True)

    db = SessionLocal()

    try:
        update_job(job_id, {
            "status": "running",
            "progress": 0.0,
            "updated_at": datetime.utcnow().isoformat(),
        })

        video = crud.get_video(db, video_id)
        if not video:
            raise RuntimeError("Video not found")

        video_path = UPLOAD_DIR / video.filename
        if not video_path.exists():
            raise RuntimeError(f"Video file missing: {video_path}")

        cap = cv2.VideoCapture(str(video_path))
        print(f"[autolabel][{job_id}] Video opened")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1

        if overwrite:
            db.query(models.Annotation).filter(
                models.Annotation.video_id == video_id
            ).delete()
            db.commit()

        if end_frame in (None, 0):
            end_frame = total_frames - 1
        else:
            end_frame = min(end_frame, total_frames - 1)

        annotations: List[schemas.AnnotationCreate] = []
        # Store previous center positions for speed calculation
        prev_positions = {}  # track_id -> (cx, cy, frame_index)


        frame_idx = 0
        processed = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_idx % 50 == 0:
               print(f"[autolabel][{job_id}] Processing frame {frame_idx}")


            if frame_idx < start_frame or frame_idx > end_frame:
                frame_idx += 1
                continue

            if (frame_idx - start_frame) % frame_stride != 0:
                frame_idx += 1
                continue

            h, w = frame.shape[:2]
            scale = 1.0
            proc_frame = frame

            if w > MAX_FRAME_WIDTH:
                scale = MAX_FRAME_WIDTH / float(w)
                proc_frame = cv2.resize(
                    frame,
                    (MAX_FRAME_WIDTH, int(h * scale)),
                    interpolation=cv2.INTER_LINEAR,
                )

            print(f"[autolabel][{job_id}] Running YOLO on frame {frame_idx}")
            results = yolo_model(proc_frame, verbose=False)[0]
            print(f"[autolabel][{job_id}] YOLO done, boxes={len(results.boxes)}")

            detections = []

            for box in results.boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])

                if conf < conf_th or cls_id not in CLASS_MAP:
                    continue

                x1, y1, x2, y2 = box.xyxy[0].tolist()

                if scale != 1.0:
                    inv = 1.0 / scale
                    x1 *= inv
                    y1 *= inv
                    x2 *= inv
                    y2 *= inv

                detections.append(
                    ([x1, y1, x2 - x1, y2 - y1], conf, CLASS_MAP[cls_id])
                )

            print(f"[autolabel][{job_id}] Running DeepSORT, detections={len(detections)}")
            tracks = tracker.update_tracks(detections, frame=proc_frame)
            print(f"[autolabel][{job_id}] DeepSORT done")
            label = trk.get_det_class() or "object"


            for trk in tracks:
                if not trk.is_confirmed():
                    continue

                l, t, r, b = trk.to_ltrb()
                track_id = int(trk.track_id)
                # Center point
                cx = (l + r) / 2.0
                cy = (t + b) / 2.0

                speed = None

                if track_id in prev_positions:
                   prev_cx, prev_cy, prev_frame = prev_positions[track_id]

                   frame_delta = frame_idx - prev_frame
                   if frame_delta > 0:
                       dx = cx - prev_cx
                       dy = cy - prev_cy
                       speed = (dx * dx + dy * dy) ** 0.5 / frame_delta
                   
                prev_positions[track_id] = (cx, cy, frame_idx)



                if scale != 1.0:
                    inv = 1.0 / scale
                    l *= inv
                    t *= inv
                    r *= inv
                    b *= inv

                annotations.append(
                   schemas.AnnotationCreate(
                       video_id=video_id,
                       frame_index=frame_idx,
                       track_id=track_id,
                       class_label=label,
                       x1=float(l),
                       y1=float(t),
                       x2=float(r),
                       y2=float(b),
                       extra_meta={
                           "speed_px_per_frame": speed
                       },
                  )
              ) 

            processed += 1
            if processed % 10 == 0:
                update_job(job_id, {
                    "progress": processed / max((end_frame - start_frame + 1), 1),
                    "updated_at": datetime.utcnow().isoformat(),
                })

            frame_idx += 1

        print(f"[autolabel][{job_id}] Finished frame loop")

        cap.release()

        if annotations:
            crud.create_annotations_bulk(db, video_id, annotations)

        update_job(job_id, {
            "status": "completed",
            "progress": 1.0,
            "updated_at": datetime.utcnow().isoformat(),
        })

        print(f"[autolabel][{job_id}] Completed with {len(annotations)} annotations")

    except Exception as e:
        print(f"[autolabel][{job_id}] FAILED:", e)
        update_job(job_id, {
            "status": "failed",
            "error_message": str(e),
            "updated_at": datetime.utcnow().isoformat(),
        })

    finally:
        db.close()

# ===============================
# WORKER LOOP
# ===============================
def _worker_loop():
    print("[autolabel] Worker loop started.")
    while True:
        for job in list(job_store.values()):
            if job["status"] == "queued":
                _run_single_job(job["job_id"])
        time.sleep(2)

# ===============================
# STARTUP ENTRYPOINT
# ===============================
def start_worker():
    threading.Thread(target=_worker_loop, daemon=True).start()
    print("[autolabel] Background worker thread started.")
