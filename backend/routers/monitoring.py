"""
Monitoring Router - Real-time system monitoring and validation APIs.
"""

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from typing import Dict, List, Any
from datetime import datetime

from database import get_db
import models
from monitoring import get_memory_usage, get_gpu_memory_usage
from metrics import Detection, calculate_iou

router = APIRouter(
    prefix="/monitoring",
    tags=["monitoring"],
)

# Global stats (shared across requests)
_system_stats = {
    "start_time": datetime.utcnow().isoformat(),
    "total_jobs_completed": 0,
    "total_frames_processed": 0,
    "total_detections": 0,
    "total_id_switches": 0,
}


@router.get("/health")
def health_check():
    """System health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "uptime_since": _system_stats["start_time"],
    }


@router.get("/metrics")
def get_metrics(db: Session = Depends(get_db)):
    """
    Real-time system metrics for monitoring dashboard.
    """
    # Database stats
    video_count = db.query(models.Video).count()
    annotation_count = db.query(models.Annotation).count()
    
    # Unique tracks
    track_count = db.query(models.Annotation.track_id).distinct().count()
    
    # Memory usage
    memory = get_memory_usage()
    gpu_memory = get_gpu_memory_usage()
    
    return {
        "timestamp": datetime.utcnow().isoformat(),
        "database": {
            "videos": video_count,
            "annotations": annotation_count,
            "unique_tracks": track_count,
        },
        "system": {
            "memory": memory,
            "gpu": gpu_memory,
        },
        "processing": _system_stats,
    }


@router.get("/validate/{video_id}")
def validate_annotations(video_id: int, db: Session = Depends(get_db)):
    """
    Validate annotation quality for a video.
    
    Checks:
    - Bounding box coordinates are valid
    - No duplicate track IDs per frame
    - Confidence scores in valid range
    - Temporal track consistency
    """
    # Check video exists
    video = db.query(models.Video).filter(models.Video.id == video_id).first()
    if not video:
        raise HTTPException(status_code=404, detail="Video not found")
    
    # Fetch annotations
    annotations = db.query(models.Annotation).filter(
        models.Annotation.video_id == video_id
    ).order_by(models.Annotation.frame_index).all()
    
    if not annotations:
        return {"video_id": video_id, "status": "no_annotations", "issues": []}
    
    issues: List[Dict] = []
    
    # Group by frame
    frames: Dict[int, List] = {}
    for ann in annotations:
        if ann.frame_index not in frames:
            frames[ann.frame_index] = []
        frames[ann.frame_index].append(ann)
    
    # Validation checks
    for frame_idx, frame_anns in frames.items():
        # Check for duplicate track IDs in same frame
        track_ids = [a.track_id for a in frame_anns if a.track_id is not None]
        if len(track_ids) != len(set(track_ids)):
            issues.append({
                "type": "duplicate_track_id",
                "frame": frame_idx,
                "severity": "high",
            })
        
        for ann in frame_anns:
            # Check bounding box validity
            if ann.x1 < 0 or ann.y1 < 0:
                issues.append({
                    "type": "negative_coordinates",
                    "frame": frame_idx,
                    "annotation_id": ann.id,
                    "severity": "high",
                })
            
            if ann.x2 <= ann.x1 or ann.y2 <= ann.y1:
                issues.append({
                    "type": "invalid_bbox_dimensions",
                    "frame": frame_idx,
                    "annotation_id": ann.id,
                    "severity": "high",
                })
            
            # Check confidence if present
            if ann.extra_meta and "confidence" in ann.extra_meta:
                conf = ann.extra_meta["confidence"]
                if not (0.0 <= conf <= 1.0):
                    issues.append({
                        "type": "invalid_confidence",
                        "frame": frame_idx,
                        "annotation_id": ann.id,
                        "severity": "medium",
                    })
    
    # Track continuity check
    track_frames: Dict[int, List[int]] = {}
    for ann in annotations:
        if ann.track_id is not None:
            if ann.track_id not in track_frames:
                track_frames[ann.track_id] = []
            track_frames[ann.track_id].append(ann.frame_index)
    
    for track_id, frame_list in track_frames.items():
        frame_list.sort()
        # Check for large gaps (potential fragmentation)
        for i in range(1, len(frame_list)):
            gap = frame_list[i] - frame_list[i-1]
            if gap > 30:  # More than 30 frames gap
                issues.append({
                    "type": "track_fragmentation",
                    "track_id": track_id,
                    "gap_start": frame_list[i-1],
                    "gap_end": frame_list[i],
                    "severity": "low",
                })
    
    # Summary
    high_severity = len([i for i in issues if i["severity"] == "high"])
    medium_severity = len([i for i in issues if i["severity"] == "medium"])
    low_severity = len([i for i in issues if i["severity"] == "low"])
    
    return {
        "video_id": video_id,
        "status": "valid" if not issues else "issues_found",
        "summary": {
            "total_annotations": len(annotations),
            "total_frames": len(frames),
            "total_tracks": len(track_frames),
            "issues_high": high_severity,
            "issues_medium": medium_severity,
            "issues_low": low_severity,
        },
        "issues": issues[:50],  # Limit to first 50 issues
    }


@router.get("/stats/{video_id}")
def get_video_stats(video_id: int, db: Session = Depends(get_db)):
    """
    Get detailed statistics for a video's annotations.
    """
    video = db.query(models.Video).filter(models.Video.id == video_id).first()
    if not video:
        raise HTTPException(status_code=404, detail="Video not found")
    
    annotations = db.query(models.Annotation).filter(
        models.Annotation.video_id == video_id
    ).all()
    
    # Class distribution
    class_counts: Dict[str, int] = {}
    for ann in annotations:
        class_counts[ann.class_label] = class_counts.get(ann.class_label, 0) + 1
    
    # Track stats
    track_ids = set(a.track_id for a in annotations if a.track_id is not None)
    
    # Frame coverage
    frames = set(a.frame_index for a in annotations)
    
    return {
        "video_id": video_id,
        "video_filename": video.original_filename,
        "total_annotations": len(annotations),
        "unique_tracks": len(track_ids),
        "frames_with_annotations": len(frames),
        "class_distribution": class_counts,
        "annotations_per_frame": round(len(annotations) / max(len(frames), 1), 2),
    }


def update_system_stats(frames: int = 0, detections: int = 0, id_switches: int = 0):
    """Update global system stats (called from worker)."""
    _system_stats["total_frames_processed"] += frames
    _system_stats["total_detections"] += detections
    _system_stats["total_id_switches"] += id_switches


def increment_jobs_completed():
    """Increment completed jobs counter."""
    _system_stats["total_jobs_completed"] += 1
