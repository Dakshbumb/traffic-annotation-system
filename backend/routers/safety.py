from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from datetime import datetime

from database import get_db
import models
from pydantic import BaseModel
from collision_detector import detect_near_misses, DetectedNearMiss

router = APIRouter(
    prefix="/api/safety",
    tags=["safety"],
)

# ------------------- SCHEMAS -------------------

class NearMissEventResponse(BaseModel):
    id: int
    video_id: int
    frame_index: int
    track_id_1: int
    track_id_2: int
    ttc: float
    distance: float
    relative_speed: float
    severity: str
    
    class Config:
        from_attributes = True

class SafetyAnalysisResponse(BaseModel):
    events_detected: int
    critical_events: int
    events: List[dict]

# ------------------- ENDPOINTS -------------------

@router.post("/analyze/{video_id}", response_model=SafetyAnalysisResponse)
def analyze_near_misses(
    video_id: int,
    fps: float = 30.0,
    db: Session = Depends(get_db)
):
    """
    Analyze video for near-miss/collision risk events.
    Calculates Time-To-Collision (TTC) for all tracked vehicles.
    """
    # Verify video
    video = db.query(models.Video).filter(models.Video.id == video_id).first()
    if not video:
        raise HTTPException(status_code=404, detail="Video not found")
        
    # fetch all annotations with track_id
    annotations = db.query(models.Annotation).filter(
        models.Annotation.video_id == video_id,
        models.Annotation.track_id.isnot(None)
    ).all()
    
    if not annotations:
        return SafetyAnalysisResponse(events_detected=0, critical_events=0, events=[])
        
    # Group by track_id
    tracks = {}
    for ann in annotations:
        if ann.track_id not in tracks:
            tracks[ann.track_id] = []
        tracks[ann.track_id].append({
            "frame_index": ann.frame_index,
            "bbox": [ann.x1, ann.y1, ann.x2, ann.y2]
        })
        
    # Run detection
    try:
        detected_events = detect_near_misses(tracks, fps)
        
        # Save to DB
        # First clear existing for this video to avoid duplicates on re-run
        db.query(models.NearMissEvent).filter(models.NearMissEvent.video_id == video_id).delete()
        
        db_events = []
        critical_count = 0
        
        for evt in detected_events:
            if evt.severity == "critical":
                critical_count += 1
                
            db_events.append(models.NearMissEvent(
                video_id=video_id,
                frame_index=evt.frame_index,
                track_id_1=evt.track_id_1,
                track_id_2=evt.track_id_2,
                ttc=evt.ttc,
                distance=evt.distance,
                relative_speed=evt.relative_speed,
                severity=evt.severity
            ))
            
        db.add_all(db_events)
        db.commit()
        
        return SafetyAnalysisResponse(
            events_detected=len(db_events),
            critical_events=critical_count,
            events=[{
                "frame": e.frame_index,
                "ttc": e.ttc,
                "severity": e.severity
            } for e in db_events]
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")


@router.get("/events/{video_id}", response_model=List[NearMissEventResponse])
def get_safety_events(
    video_id: int,
    severity: Optional[str] = None,
    db: Session = Depends(get_db)
):
    """Get detected safety events for a video."""
    query = db.query(models.NearMissEvent).filter(models.NearMissEvent.video_id == video_id)
    
    if severity:
        query = query.filter(models.NearMissEvent.severity == severity)
        
    events = query.order_by(models.NearMissEvent.frame_index).all()
    return events


@router.delete("/events/{video_id}")
def clear_safety_events(
    video_id: int,
    db: Session = Depends(get_db)
):
    """Clear all safety events for a video."""
    deleted = db.query(models.NearMissEvent).filter(models.NearMissEvent.video_id == video_id).delete()
    db.commit()
    return {"message": f"Deleted {deleted} events"}
