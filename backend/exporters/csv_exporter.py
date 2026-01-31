"""
CSV Exporter
Exports annotations as flat CSV for analytics and spreadsheet use.
"""

import csv
import os
from typing import Dict, List
from sqlalchemy.orm import Session

import models
from export_schemas import get_class_id


def export_csv(
    db: Session,
    video_id: int,
    video_filename: str,
    frame_width: int,
    frame_height: int,
    fps: float,
    output_path: str,
) -> Dict[str, any]:
    """
    Export annotations as flat CSV.
    
    Columns:
    frame_id, timestamp, track_id, class, class_id, confidence,
    x1, y1, x2, y2, cx, cy, width, height, area
    
    Returns:
        Dict with export stats
    """
    # Fetch all annotations
    annotations = db.query(models.Annotation).filter(
        models.Annotation.video_id == video_id
    ).order_by(models.Annotation.frame_index).all()

    # Write CSV
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
    
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        
        # Header
        writer.writerow([
            "video_id", "video_filename", "frame_id", "timestamp",
            "track_id", "class", "class_id", "confidence",
            "x1", "y1", "x2", "y2", "cx", "cy", "width", "height", "area"
        ])
        
        for ann in annotations:
            x1, y1, x2, y2 = ann.x1, ann.y1, ann.x2, ann.y2
            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2
            width = x2 - x1
            height = y2 - y1
            area = width * height
            
            conf = 1.0
            if ann.extra_meta and "confidence" in ann.extra_meta:
                conf = ann.extra_meta["confidence"]
            
            timestamp = ann.frame_index / fps if fps > 0 else 0.0
            
            writer.writerow([
                video_id,
                video_filename,
                ann.frame_index,
                round(timestamp, 4),
                ann.track_id if ann.track_id is not None else -1,
                ann.class_label,
                get_class_id(ann.class_label),
                round(conf, 4),
                round(x1, 2), round(y1, 2), round(x2, 2), round(y2, 2),
                round(cx, 2), round(cy, 2),
                round(width, 2), round(height, 2), round(area, 2),
            ])

    return {
        "total_annotations": len(annotations),
        "output_path": output_path,
    }


def export_analytics_csv(
    db: Session,
    video_id: int,
    output_path: str,
) -> Dict[str, any]:
    """
    Export analytics line data as CSV.
    
    Returns:
        Dict with export stats
    """
    # Fetch analytics lines
    lines = db.query(models.AnalyticsLine).filter(
        models.AnalyticsLine.video_id == video_id
    ).all()

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        
        # Header
        writer.writerow([
            "line_id", "name", "line_type", "point1_x", "point1_y",
            "point2_x", "point2_y", "count_in", "count_out", "total"
        ])
        
        for line in lines:
            points = line.points or [[0, 0], [0, 0]]
            meta = line.meta_data or {}
            
            count_in = meta.get("in", 0)
            count_out = meta.get("out", 0)
            
            writer.writerow([
                line.id,
                line.name,
                line.line_type,
                points[0][0] if len(points) > 0 else 0,
                points[0][1] if len(points) > 0 else 0,
                points[1][0] if len(points) > 1 else 0,
                points[1][1] if len(points) > 1 else 0,
                count_in,
                count_out,
                count_in + count_out,
            ])

    return {
        "total_lines": len(lines),
        "output_path": output_path,
    }
