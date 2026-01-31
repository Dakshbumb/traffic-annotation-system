"""
Custom JSON Exporter
Exports annotations in the user-specified JSON schema format.
"""

import json
from datetime import datetime
from typing import List, Dict, Any
from sqlalchemy.orm import Session

import models
from export_schemas import (
    FrameExport, DetectionExport, FrameMetadata, 
    DetectionAttributes, VideoExport, get_class_id
)


def export_custom_json(
    db: Session,
    video_id: int,
    video: models.Video,
    frame_width: int,
    frame_height: int,
    fps: float,
) -> Dict[str, Any]:
    """
    Export annotations in the custom JSON schema format.
    
    Returns:
        Dict matching the user-specified JSON schema
    """
    # Fetch all annotations
    annotations = db.query(models.Annotation).filter(
        models.Annotation.video_id == video_id
    ).order_by(models.Annotation.frame_index).all()

    # Group by frame
    frames_data: Dict[int, List[models.Annotation]] = {}
    class_counts: Dict[str, int] = {}
    
    for ann in annotations:
        frame_idx = ann.frame_index
        if frame_idx not in frames_data:
            frames_data[frame_idx] = []
        frames_data[frame_idx].append(ann)
        
        # Count classes
        label = ann.class_label
        class_counts[label] = class_counts.get(label, 0) + 1

    # Build frames
    frames: List[Dict] = []
    
    for frame_idx in sorted(frames_data.keys()):
        frame_anns = frames_data[frame_idx]
        
        detections = []
        for ann in frame_anns:
            x1, y1, x2, y2 = ann.x1, ann.y1, ann.x2, ann.y2
            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2
            area = (x2 - x1) * (y2 - y1)
            
            conf = 1.0
            if ann.extra_meta and "confidence" in ann.extra_meta:
                conf = ann.extra_meta["confidence"]
            
            detection = {
                "track_id": ann.track_id if ann.track_id is not None else -1,
                "class": ann.class_label,
                "class_id": get_class_id(ann.class_label),
                "confidence": round(conf, 4),
                "bbox": [round(x1, 2), round(y1, 2), round(x2, 2), round(y2, 2)],
                "bbox_center": [round(cx, 2), round(cy, 2)],
                "area": round(area, 2),
                "attributes": {
                    "occluded": False,
                    "truncated": False,
                    "difficult": False,
                }
            }
            detections.append(detection)
        
        # Calculate timestamp from frame index and fps
        timestamp = frame_idx / fps if fps > 0 else 0.0
        
        frame_export = {
            "frame_id": frame_idx,
            "timestamp": round(timestamp, 4),
            "camera_id": "default",
            "detections": detections,
            "metadata": {
                "frame_width": frame_width,
                "frame_height": frame_height,
                "fps": fps,
                "processing_time_ms": None,
            }
        }
        frames.append(frame_export)

    # Build final export
    export_data = {
        "video_id": video_id,
        "video_filename": video.original_filename,
        "export_format": "custom_json",
        "export_timestamp": datetime.utcnow().isoformat(),
        "total_frames": len(frames),
        "total_detections": len(annotations),
        "classes": class_counts,
        "frames": frames,
    }
    
    return export_data


def save_custom_json(data: Dict, output_path: str) -> str:
    """Save custom JSON data to file."""
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    return output_path
