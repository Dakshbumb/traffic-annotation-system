"""
COCO JSON Exporter
Exports annotations in COCO format for object detection.
"""

import json
from datetime import datetime
from typing import List, Dict, Any
from sqlalchemy.orm import Session

import models
from export_schemas import CLASS_ID_MAP, get_class_id


def export_coco_json(
    db: Session,
    video_id: int,
    video: models.Video,
    frame_width: int,
    frame_height: int,
    fps: float,
) -> Dict[str, Any]:
    """
    Export annotations in COCO JSON format.
    
    Returns:
        Dict containing COCO-format annotations
    """
    # Fetch all annotations
    annotations = db.query(models.Annotation).filter(
        models.Annotation.video_id == video_id
    ).order_by(models.Annotation.frame_index).all()

    # Build COCO structure
    coco_data = {
        "info": {
            "description": f"Traffic annotations for {video.original_filename}",
            "url": "",
            "version": "1.0",
            "year": datetime.utcnow().year,
            "contributor": "Smart Traffic Annotation System",
            "date_created": datetime.utcnow().isoformat(),
        },
        "licenses": [
            {"id": 1, "name": "Unknown", "url": ""}
        ],
        "categories": [
            {"id": v, "name": k, "supercategory": "vehicle" if k != "person" else "person"}
            for k, v in CLASS_ID_MAP.items()
        ],
        "images": [],
        "annotations": [],
    }

    # Track unique frames
    frame_ids = set()
    annotation_id = 1

    for ann in annotations:
        frame_idx = ann.frame_index

        # Add image entry if not exists
        if frame_idx not in frame_ids:
            frame_ids.add(frame_idx)
            coco_data["images"].append({
                "id": frame_idx,
                "file_name": f"frame_{frame_idx:06d}.jpg",
                "width": frame_width,
                "height": frame_height,
            })

        # Calculate bbox in COCO format [x, y, width, height]
        x1, y1, x2, y2 = ann.x1, ann.y1, ann.x2, ann.y2
        width = x2 - x1
        height = y2 - y1
        area = width * height

        class_id = get_class_id(ann.class_label)
        if class_id < 0:
            continue

        coco_data["annotations"].append({
            "id": annotation_id,
            "image_id": frame_idx,
            "category_id": class_id,
            "bbox": [x1, y1, width, height],
            "area": area,
            "iscrowd": 0,
            "attributes": {
                "track_id": ann.track_id,
                "confidence": ann.extra_meta.get("confidence", 1.0) if ann.extra_meta else 1.0,
            }
        })
        annotation_id += 1

    return coco_data


def save_coco_json(coco_data: Dict, output_path: str) -> str:
    """Save COCO data to JSON file."""
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(coco_data, f, indent=2)
    return output_path
