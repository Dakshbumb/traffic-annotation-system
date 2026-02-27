"""
Export Router - API endpoints for all export formats.
Supports: COCO JSON, YOLO TXT, Pascal VOC XML, Custom JSON, CSV
"""

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from sqlalchemy.orm import Session

from datetime import datetime
from typing import Dict
import os
import shutil
import cv2

from database import get_db
import models
from config import UPLOAD_DIR, EXPORT_DIR

# Import exporters
from exporters.coco_exporter import export_coco_json, save_coco_json
from exporters.yolo_exporter import export_yolo_txt
from exporters.voc_exporter import export_voc_xml
from exporters.json_exporter import export_custom_json, save_custom_json
from exporters.csv_exporter import export_csv, export_analytics_csv
from export_schemas import CLASS_ID_MAP

router = APIRouter(
    prefix="/export",
    tags=["export"],
)

# Ensure export dir exists
os.makedirs(EXPORT_DIR, exist_ok=True)


def get_video_metadata(video_path: str) -> Dict:
    """Get video dimensions and FPS from file."""
    if not os.path.exists(video_path):
        return {"width": 1920, "height": 1080, "fps": 30.0}
    
    cap = cv2.VideoCapture(str(video_path))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1920
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 1080
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    cap.release()
    
    return {"width": width, "height": height, "fps": fps}


# ------------------- COCO JSON -------------------
@router.get("/coco/{video_id}")
def export_coco(video_id: int, db: Session = Depends(get_db)):
    """Export annotations in COCO JSON format."""
    video = db.query(models.Video).filter(models.Video.id == video_id).first()
    if not video:
        raise HTTPException(status_code=404, detail="Video not found")
    
    video_path = os.path.join(UPLOAD_DIR, video.filename)
    meta = get_video_metadata(video_path)
    
    coco_data = export_coco_json(
        db, video_id, video, meta["width"], meta["height"], meta["fps"]
    )
    
    # Save to file
    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    filename = f"video_{video_id}_coco_{ts}.json"
    filepath = os.path.join(EXPORT_DIR, filename)
    save_coco_json(coco_data, filepath)
    
    return FileResponse(
        path=filepath,
        media_type="application/json",
        filename=filename,
    )


# ------------------- YOLO TXT -------------------
@router.get("/yolo/{video_id}")
def export_yolo(video_id: int, db: Session = Depends(get_db)):
    """Export annotations in YOLO TXT format (zipped)."""
    video = db.query(models.Video).filter(models.Video.id == video_id).first()
    if not video:
        raise HTTPException(status_code=404, detail="Video not found")
    
    video_path = os.path.join(UPLOAD_DIR, video.filename)
    meta = get_video_metadata(video_path)
    
    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    folder_name = f"video_{video_id}_yolo_{ts}"
    folder_path = os.path.join(EXPORT_DIR, folder_name)
    
    export_yolo_txt(db, video_id, meta["width"], meta["height"], folder_path)
    
    # Write classes.txt
    classes_file = os.path.join(folder_path, "classes.txt")
    with open(classes_file, "w") as f:
        for class_name, class_id in sorted(CLASS_ID_MAP.items(), key=lambda x: x[1]):
            f.write(f"{class_name}\n")
    
    # ZIP
    zip_path = shutil.make_archive(folder_path, "zip", folder_path)
    
    return FileResponse(
        path=zip_path,
        media_type="application/zip",
        filename=f"{folder_name}.zip",
    )


# ------------------- Pascal VOC XML -------------------
@router.get("/voc/{video_id}")
def export_voc(video_id: int, db: Session = Depends(get_db)):
    """Export annotations in Pascal VOC XML format (zipped)."""
    video = db.query(models.Video).filter(models.Video.id == video_id).first()
    if not video:
        raise HTTPException(status_code=404, detail="Video not found")
    
    video_path = os.path.join(UPLOAD_DIR, video.filename)
    meta = get_video_metadata(video_path)
    
    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    folder_name = f"video_{video_id}_voc_{ts}"
    folder_path = os.path.join(EXPORT_DIR, folder_name)
    
    export_voc_xml(
        db, video_id, video.original_filename,
        meta["width"], meta["height"], folder_path
    )
    
    # ZIP
    zip_path = shutil.make_archive(folder_path, "zip", folder_path)
    
    return FileResponse(
        path=zip_path,
        media_type="application/zip",
        filename=f"{folder_name}.zip",
    )


# ------------------- Custom JSON -------------------
@router.get("/json/{video_id}")
def export_json(video_id: int, db: Session = Depends(get_db)):
    """Export annotations in custom JSON schema format."""
    video = db.query(models.Video).filter(models.Video.id == video_id).first()
    if not video:
        raise HTTPException(status_code=404, detail="Video not found")
    
    video_path = os.path.join(UPLOAD_DIR, video.filename)
    meta = get_video_metadata(video_path)
    
    json_data = export_custom_json(
        db, video_id, video, meta["width"], meta["height"], meta["fps"]
    )
    
    # Save to file
    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    filename = f"video_{video_id}_annotations_{ts}.json"
    filepath = os.path.join(EXPORT_DIR, filename)
    save_custom_json(json_data, filepath)
    
    return FileResponse(
        path=filepath,
        media_type="application/json",
        filename=filename,
    )


# ------------------- CSV -------------------
@router.get("/csv/{video_id}")
def export_csv_endpoint(video_id: int, db: Session = Depends(get_db)):
    """Export annotations as flat CSV."""
    video = db.query(models.Video).filter(models.Video.id == video_id).first()
    if not video:
        raise HTTPException(status_code=404, detail="Video not found")
    
    video_path = os.path.join(UPLOAD_DIR, video.filename)
    meta = get_video_metadata(video_path)
    
    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    filename = f"video_{video_id}_annotations_{ts}.csv"
    filepath = os.path.join(EXPORT_DIR, filename)
    
    export_csv(
        db, video_id, video.original_filename,
        meta["width"], meta["height"], meta["fps"], filepath
    )
    
    return FileResponse(
        path=filepath,
        media_type="text/csv",
        filename=filename,
    )


# ------------------- Analytics CSV -------------------
@router.get("/analytics_csv/{video_id}")
def export_analytics_csv_endpoint(video_id: int, db: Session = Depends(get_db)):
    """Export analytics line counts as CSV."""
    video = db.query(models.Video).filter(models.Video.id == video_id).first()
    if not video:
        raise HTTPException(status_code=404, detail="Video not found")
    
    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    filename = f"video_{video_id}_analytics_{ts}.csv"
    filepath = os.path.join(EXPORT_DIR, filename)
    
    export_analytics_csv(db, video_id, filepath)
    
    return FileResponse(
        path=filepath,
        media_type="text/csv",
        filename=filename,
    )


# ------------------- Legacy YOLO-MOT (keep for backwards compatibility) -------------------
@router.get("/yolo_mot/{video_id}")
def export_yolo_mot(video_id: int, db: Session = Depends(get_db)):
    """Legacy: Export in YOLO-MOT style (frame, track_id, x, y, w, h, conf, class, vis)."""
    import csv
    
    video = db.query(models.Video).filter(models.Video.id == video_id).first()
    if not video:
        raise HTTPException(status_code=404, detail="Video not found")

    anns = db.query(models.Annotation).filter(
        models.Annotation.video_id == video_id
    ).order_by(models.Annotation.frame_index.asc()).all()
    
    if not anns:
        raise HTTPException(status_code=400, detail="No annotations for this video")

    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    folder_name = f"video_{video_id}_yolo_mot_{ts}"
    folder_path = os.path.join(EXPORT_DIR, folder_name)
    os.makedirs(folder_path, exist_ok=True)

    annotations_txt = os.path.join(folder_path, "annotations.txt")

    with open(annotations_txt, "w", newline="") as f:
        writer = csv.writer(f)
        for ann in anns:
            class_id = CLASS_ID_MAP.get(ann.class_label, -1)
            if class_id < 0:
                continue
            frame = ann.frame_index
            track_id = ann.track_id if ann.track_id is not None else -1
            x, y = ann.x1, ann.y1
            w, h = max(0, ann.x2 - ann.x1), max(0, ann.y2 - ann.y1)
            conf, visibility = 1.0, 1.0
            writer.writerow([frame, track_id, x, y, w, h, conf, class_id, visibility])

    zip_path = shutil.make_archive(folder_path, "zip", folder_path)
    
    return FileResponse(
        path=zip_path,
        media_type="application/zip",
        filename=f"{folder_name}.zip",
    )


# ------------------- Annotated Video (MP4 with bboxes) -------------------
@router.get("/annotated_video/{video_id}")
def export_annotated_video(video_id: int, db: Session = Depends(get_db)):
    """Export video with bounding boxes drawn on each frame."""
    video = db.query(models.Video).filter(models.Video.id == video_id).first()
    if not video:
        raise HTTPException(status_code=404, detail="Video not found")

    video_path = os.path.join(UPLOAD_DIR, video.filename)
    if not os.path.exists(video_path):
        raise HTTPException(status_code=404, detail="Video file not found on disk")

    # Get all annotations grouped by frame
    anns = db.query(models.Annotation).filter(
        models.Annotation.video_id == video_id
    ).order_by(models.Annotation.frame_index.asc()).all()

    if not anns:
        raise HTTPException(status_code=400, detail="No annotations found for this video")

    # Group annotations by frame
    from collections import defaultdict
    frame_anns = defaultdict(list)
    for a in anns:
        frame_anns[a.frame_index].append(a)

    # Class colors (BGR for OpenCV)
    CLASS_COLORS_BGR = {
        'car': (94, 197, 34),       # green
        'truck': (22, 115, 249),    # orange
        'bus': (8, 179, 234),       # yellow
        'motorcycle': (212, 182, 6),# cyan
        'bicycle': (239, 70, 217),  # magenta
        'person': (68, 68, 239),    # red
    }

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise HTTPException(status_code=500, detail="Cannot open video")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    out_filename = f"annotated_video_{video_id}_{ts}.mp4"
    out_path = os.path.join(EXPORT_DIR, out_filename)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(out_path, fourcc, fps, (width, height))

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Draw annotations for this frame
        if frame_idx in frame_anns:
            for ann in frame_anns[frame_idx]:
                color = CLASS_COLORS_BGR.get(ann.class_label, (255, 255, 255))
                x1, y1 = int(ann.x1), int(ann.y1)
                x2, y2 = int(ann.x2), int(ann.y2)

                # Draw box
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

                # Draw label background
                label = f"{ann.class_label}"
                if ann.track_id is not None:
                    label += f" #{ann.track_id}"
                if ann.extra_meta and ann.extra_meta.get('speed_kmh'):
                    label += f" {ann.extra_meta['speed_kmh']}km/h"

                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                cv2.rectangle(frame, (x1, y1 - th - 8), (x1 + tw + 6, y1), color, -1)
                cv2.putText(frame, label, (x1 + 3, y1 - 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

        writer.write(frame)
        frame_idx += 1

    cap.release()
    writer.release()

    return FileResponse(
        path=out_path,
        media_type="video/mp4",
        filename=out_filename,
    )
