"""
YOLO TXT Exporter
Exports annotations in YOLO format (one .txt per frame).
Format: class_id cx cy w h (normalized 0-1)
"""

import os
from typing import List, Dict
from sqlalchemy.orm import Session

import models
from export_schemas import get_class_id


def export_yolo_txt(
    db: Session,
    video_id: int,
    frame_width: int,
    frame_height: int,
    output_dir: str,
) -> Dict[str, any]:
    """
    Export annotations in YOLO TXT format.
    Creates one .txt file per frame with annotations.
    
    Format per line: class_id cx cy w h (normalized)
    
    Returns:
        Dict with export stats
    """
    # Fetch all annotations
    annotations = db.query(models.Annotation).filter(
        models.Annotation.video_id == video_id
    ).order_by(models.Annotation.frame_index).all()

    # Group by frame
    frames_data: Dict[int, List[str]] = {}
    
    for ann in annotations:
        frame_idx = ann.frame_index
        
        class_id = get_class_id(ann.class_label)
        if class_id < 0:
            continue

        # Convert to YOLO format (normalized center + w/h)
        x1, y1, x2, y2 = ann.x1, ann.y1, ann.x2, ann.y2
        
        # Center coordinates
        cx = ((x1 + x2) / 2) / frame_width
        cy = ((y1 + y2) / 2) / frame_height
        
        # Width and height (normalized)
        w = (x2 - x1) / frame_width
        h = (y2 - y1) / frame_height
        
        # Clamp to [0, 1]
        cx = max(0, min(1, cx))
        cy = max(0, min(1, cy))
        w = max(0, min(1, w))
        h = max(0, min(1, h))
        
        line = f"{class_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}"
        
        if frame_idx not in frames_data:
            frames_data[frame_idx] = []
        frames_data[frame_idx].append(line)

    # Write files
    os.makedirs(output_dir, exist_ok=True)
    files_created = 0
    
    for frame_idx, lines in frames_data.items():
        filename = f"frame_{frame_idx:06d}.txt"
        filepath = os.path.join(output_dir, filename)
        
        with open(filepath, "w") as f:
            f.write("\n".join(lines))
        files_created += 1

    # Write classes.txt
    classes_file = os.path.join(output_dir, "classes.txt")
    with open(classes_file, "w") as f:
        for class_name in sorted(get_class_id.__self__ if hasattr(get_class_id, '__self__') else {}, key=lambda x: get_class_id(x)):
            f.write(f"{class_name}\n")

    return {
        "files_created": files_created,
        "total_annotations": len(annotations),
        "output_dir": output_dir,
    }
