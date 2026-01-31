"""
Pascal VOC XML Exporter
Exports annotations in Pascal VOC format (one .xml per frame).
"""

import os
from xml.etree.ElementTree import Element, SubElement, ElementTree
from typing import Dict, List
from sqlalchemy.orm import Session

import models


def create_voc_xml(
    frame_idx: int,
    annotations: List[models.Annotation],
    frame_width: int,
    frame_height: int,
    video_filename: str,
) -> Element:
    """Create a Pascal VOC XML element for a single frame."""
    
    root = Element("annotation")
    
    # Folder
    SubElement(root, "folder").text = "frames"
    
    # Filename
    SubElement(root, "filename").text = f"frame_{frame_idx:06d}.jpg"
    
    # Source
    source = SubElement(root, "source")
    SubElement(source, "database").text = "Smart Traffic Annotation System"
    SubElement(source, "annotation").text = video_filename
    SubElement(source, "image").text = "video_frame"
    
    # Size
    size = SubElement(root, "size")
    SubElement(size, "width").text = str(frame_width)
    SubElement(size, "height").text = str(frame_height)
    SubElement(size, "depth").text = "3"
    
    # Segmented
    SubElement(root, "segmented").text = "0"
    
    # Objects
    for ann in annotations:
        obj = SubElement(root, "object")
        SubElement(obj, "name").text = ann.class_label
        SubElement(obj, "pose").text = "Unspecified"
        SubElement(obj, "truncated").text = "0"
        SubElement(obj, "difficult").text = "0"
        
        # Optional: track_id as attribute
        if ann.track_id is not None:
            SubElement(obj, "track_id").text = str(ann.track_id)
        
        # Bounding box
        bndbox = SubElement(obj, "bndbox")
        SubElement(bndbox, "xmin").text = str(int(max(0, ann.x1)))
        SubElement(bndbox, "ymin").text = str(int(max(0, ann.y1)))
        SubElement(bndbox, "xmax").text = str(int(min(frame_width, ann.x2)))
        SubElement(bndbox, "ymax").text = str(int(min(frame_height, ann.y2)))
    
    return root


def export_voc_xml(
    db: Session,
    video_id: int,
    video_filename: str,
    frame_width: int,
    frame_height: int,
    output_dir: str,
) -> Dict[str, any]:
    """
    Export annotations in Pascal VOC XML format.
    Creates one .xml file per frame.
    
    Returns:
        Dict with export stats
    """
    # Fetch all annotations
    annotations = db.query(models.Annotation).filter(
        models.Annotation.video_id == video_id
    ).order_by(models.Annotation.frame_index).all()

    # Group by frame
    frames_data: Dict[int, List[models.Annotation]] = {}
    
    for ann in annotations:
        frame_idx = ann.frame_index
        if frame_idx not in frames_data:
            frames_data[frame_idx] = []
        frames_data[frame_idx].append(ann)

    # Write XML files
    os.makedirs(output_dir, exist_ok=True)
    files_created = 0
    
    for frame_idx, frame_anns in frames_data.items():
        xml_root = create_voc_xml(
            frame_idx, frame_anns, frame_width, frame_height, video_filename
        )
        
        filename = f"frame_{frame_idx:06d}.xml"
        filepath = os.path.join(output_dir, filename)
        
        tree = ElementTree(xml_root)
        tree.write(filepath, encoding="utf-8", xml_declaration=True)
        files_created += 1

    return {
        "files_created": files_created,
        "total_annotations": len(annotations),
        "output_dir": output_dir,
    }
