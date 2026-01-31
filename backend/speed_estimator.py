"""
Speed Estimation Module

Calculates vehicle speeds using calibration lines with known real-world distances.
"""

import math
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
import logging

logger = logging.getLogger("speed_estimator")

# Default video FPS if not specified
DEFAULT_FPS = 30.0


@dataclass
class CalibrationData:
    """Stores calibration line data."""
    pixel_distance: float
    real_distance_meters: float
    pixels_per_meter: float


def calculate_pixel_distance(p1: List[float], p2: List[float]) -> float:
    """Calculate Euclidean distance between two points in pixels."""
    return math.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)


def get_calibration(line_points: List[List[float]], real_distance_meters: float) -> CalibrationData:
    """
    Create calibration data from a line and its real-world distance.
    
    Args:
        line_points: [[x1, y1], [x2, y2]] - the calibration line endpoints
        real_distance_meters: Real-world distance in meters
    
    Returns:
        CalibrationData with pixels-per-meter ratio
    """
    if len(line_points) != 2:
        raise ValueError("Calibration line must have exactly 2 points")
    
    pixel_distance = calculate_pixel_distance(line_points[0], line_points[1])
    pixels_per_meter = pixel_distance / real_distance_meters
    
    return CalibrationData(
        pixel_distance=pixel_distance,
        real_distance_meters=real_distance_meters,
        pixels_per_meter=pixels_per_meter
    )


def get_bbox_center(bbox: List[float]) -> Tuple[float, float]:
    """Get center point of bounding box [x1, y1, x2, y2]."""
    return ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)


def calculate_track_speeds(
    track_detections: List[Dict],
    pixels_per_meter: float,
    fps: float = DEFAULT_FPS,
    frame_stride: int = 1
) -> List[Dict]:
    """
    Calculate speed for each detection in a track.
    
    Args:
        track_detections: List of {"frame_index": int, "bbox": [x1,y1,x2,y2]}
        pixels_per_meter: Calibration ratio
        fps: Video frames per second
        frame_stride: How many frames were skipped during detection
    
    Returns:
        List of {"frame_index": int, "speed_kmh": float}
    """
    if len(track_detections) < 2:
        return []
    
    # Sort by frame
    sorted_dets = sorted(track_detections, key=lambda d: d["frame_index"])
    
    speeds = []
    
    for i in range(1, len(sorted_dets)):
        prev = sorted_dets[i - 1]
        curr = sorted_dets[i]
        
        # Get centers
        prev_center = get_bbox_center(prev["bbox"])
        curr_center = get_bbox_center(curr["bbox"])
        
        # Pixel displacement
        pixel_dist = calculate_pixel_distance(
            [prev_center[0], prev_center[1]],
            [curr_center[0], curr_center[1]]
        )
        
        # Convert to meters
        meters = pixel_dist / pixels_per_meter
        
        # Time between frames (accounting for stride)
        frame_diff = curr["frame_index"] - prev["frame_index"]
        if frame_diff <= 0:
            continue
            
        time_seconds = frame_diff / fps
        
        # Speed in m/s
        speed_ms = meters / time_seconds
        
        # Convert to km/h
        speed_kmh = speed_ms * 3.6
        
        # Sanity check - cap at 300 km/h
        speed_kmh = min(speed_kmh, 300.0)
        
        speeds.append({
            "frame_index": curr["frame_index"],
            "speed_kmh": round(speed_kmh, 1),
            "speed_ms": round(speed_ms, 2)
        })
    
    return speeds


def estimate_video_speeds(
    db_session,
    video_id: int,
    fps: float = DEFAULT_FPS
) -> Dict[int, List[Dict]]:
    """
    Estimate speeds for all tracked vehicles in a video.
    
    Args:
        db_session: SQLAlchemy session
        video_id: Video ID
        fps: Video frame rate
    
    Returns:
        Dictionary mapping track_id to list of speed measurements
    """
    from models import Annotation, AnalyticsLine
    
    # Find calibration line
    calibration_line = db_session.query(AnalyticsLine).filter(
        AnalyticsLine.video_id == video_id,
        AnalyticsLine.line_type == "calibration"
    ).first()
    
    if not calibration_line:
        logger.warning(f"No calibration line found for video {video_id}")
        return {}
    
    # Get calibration
    meta = calibration_line.meta_data or {}
    real_distance = meta.get("distance_meters", 10.0)  # Default 10m
    
    try:
        calibration = get_calibration(calibration_line.points, real_distance)
    except Exception as e:
        logger.error(f"Calibration error: {e}")
        return {}
    
    logger.info(f"Calibration: {calibration.pixels_per_meter:.2f} pixels/meter")
    
    # Load all tracked annotations
    annotations = db_session.query(Annotation).filter(
        Annotation.video_id == video_id,
        Annotation.track_id.isnot(None)
    ).all()
    
    if not annotations:
        logger.warning(f"No tracked annotations for video {video_id}")
        return {}
    
    # Group by track_id
    tracks: Dict[int, List[Dict]] = {}
    for ann in annotations:
        if ann.track_id not in tracks:
            tracks[ann.track_id] = []
        tracks[ann.track_id].append({
            "frame_index": ann.frame_index,
            "bbox": [ann.x1, ann.y1, ann.x2, ann.y2],
            "annotation_id": ann.id
        })
    
    # Calculate speeds for each track
    all_speeds = {}
    for track_id, detections in tracks.items():
        speeds = calculate_track_speeds(detections, calibration.pixels_per_meter, fps)
        if speeds:
            all_speeds[track_id] = speeds
    
    logger.info(f"Calculated speeds for {len(all_speeds)} tracks")
    return all_speeds


def update_annotation_speeds(db_session, video_id: int, fps: float = DEFAULT_FPS) -> int:
    """
    Update annotation extra_meta with calculated speeds.
    
    Returns:
        Number of annotations updated
    """
    from models import Annotation
    
    # Get speeds
    all_speeds = estimate_video_speeds(db_session, video_id, fps)
    
    if not all_speeds:
        return 0
    
    # Build lookup: (track_id, frame_index) -> speed_kmh
    speed_lookup = {}
    for track_id, speeds in all_speeds.items():
        for s in speeds:
            speed_lookup[(track_id, s["frame_index"])] = s["speed_kmh"]
    
    # Update annotations
    updated = 0
    annotations = db_session.query(Annotation).filter(
        Annotation.video_id == video_id,
        Annotation.track_id.isnot(None)
    ).all()
    
    for ann in annotations:
        key = (ann.track_id, ann.frame_index)
        if key in speed_lookup:
            meta = ann.extra_meta or {}
            meta["speed_kmh"] = speed_lookup[key]
            ann.extra_meta = meta
            updated += 1
    
    db_session.commit()
    logger.info(f"Updated {updated} annotations with speed data")
    return updated


def get_speed_stats(db_session, video_id: int) -> Dict:
    """Get speed statistics for a video."""
    from models import Annotation
    from sqlalchemy import func
    
    # Query all annotations with speed
    annotations = db_session.query(Annotation).filter(
        Annotation.video_id == video_id,
        Annotation.extra_meta.isnot(None)
    ).all()
    
    speeds = []
    for ann in annotations:
        if ann.extra_meta and "speed_kmh" in ann.extra_meta:
            speeds.append(ann.extra_meta["speed_kmh"])
    
    if not speeds:
        return {"count": 0, "avg": 0, "max": 0, "min": 0}
    
    return {
        "count": len(speeds),
        "avg": round(sum(speeds) / len(speeds), 1),
        "max": round(max(speeds), 1),
        "min": round(min([s for s in speeds if s > 0] or [0]), 1)
    }
