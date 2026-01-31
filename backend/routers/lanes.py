"""
Lanes API Router

Endpoints for managing lane zones and detecting lane change events.
"""

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from pydantic import BaseModel
from typing import List, Optional
import logging

from database import get_db
from models import LaneZone, LaneEvent, Video
from lane_events import analyze_video_lanes

logger = logging.getLogger("lanes")
router = APIRouter(prefix="/api/lanes", tags=["lanes"])


# ─────────────────────────────────────────────────────────────
# Schemas
# ─────────────────────────────────────────────────────────────

class LaneZoneCreate(BaseModel):
    video_id: int
    name: str
    zone_type: str  # "ego", "adjacent_left", "adjacent_right"
    points: List[List[float]]  # [[x1,y1], [x2,y2], ...]
    color: Optional[str] = "#22c55e"


class LaneZoneResponse(BaseModel):
    id: int
    video_id: int
    name: str
    zone_type: str
    points: List[List[float]]
    color: str

    class Config:
        from_attributes = True


class LaneEventResponse(BaseModel):
    id: int
    video_id: int
    track_id: int
    event_type: str
    frame_index: int
    from_zone: str
    to_zone: str
    confidence: float
    bbox: List[float]

    class Config:
        from_attributes = True


class AnalyzeResponse(BaseModel):
    events_detected: int
    events: List[dict]


# ─────────────────────────────────────────────────────────────
# Lane Zone Endpoints
# ─────────────────────────────────────────────────────────────

@router.post("", response_model=LaneZoneResponse)
def create_lane_zone(zone: LaneZoneCreate, db: Session = Depends(get_db)):
    """Create a new lane zone for a video."""
    # Verify video exists
    video = db.query(Video).filter(Video.id == zone.video_id).first()
    if not video:
        raise HTTPException(status_code=404, detail="Video not found")
    
    db_zone = LaneZone(
        video_id=zone.video_id,
        name=zone.name,
        zone_type=zone.zone_type,
        points=zone.points,
        color=zone.color
    )
    db.add(db_zone)
    db.commit()
    db.refresh(db_zone)
    
    logger.info(f"Created lane zone: {zone.name} ({zone.zone_type}) for video {zone.video_id}")
    return db_zone


@router.get("/video/{video_id}", response_model=List[LaneZoneResponse])
def get_lane_zones(video_id: int, db: Session = Depends(get_db)):
    """Get all lane zones for a video."""
    zones = db.query(LaneZone).filter(LaneZone.video_id == video_id).all()
    return zones


@router.delete("/{zone_id}")
def delete_lane_zone(zone_id: int, db: Session = Depends(get_db)):
    """Delete a lane zone."""
    zone = db.query(LaneZone).filter(LaneZone.id == zone_id).first()
    if not zone:
        raise HTTPException(status_code=404, detail="Lane zone not found")
    
    db.delete(zone)
    db.commit()
    
    return {"message": "Lane zone deleted", "id": zone_id}


@router.delete("/video/{video_id}")
def delete_all_lane_zones(video_id: int, db: Session = Depends(get_db)):
    """Delete all lane zones for a video."""
    deleted = db.query(LaneZone).filter(LaneZone.video_id == video_id).delete()
    db.commit()
    
    return {"message": f"Deleted {deleted} lane zones", "video_id": video_id}


# ─────────────────────────────────────────────────────────────
# Lane Event Endpoints
# ─────────────────────────────────────────────────────────────

@router.get("/events/{video_id}", response_model=List[LaneEventResponse])
def get_lane_events(video_id: int, event_type: Optional[str] = None, db: Session = Depends(get_db)):
    """Get detected lane events for a video."""
    query = db.query(LaneEvent).filter(LaneEvent.video_id == video_id)
    
    if event_type:
        query = query.filter(LaneEvent.event_type == event_type)
    
    events = query.order_by(LaneEvent.frame_index).all()
    return events


@router.post("/analyze/{video_id}", response_model=AnalyzeResponse)
def analyze_lanes(video_id: int, db: Session = Depends(get_db)):
    """
    Analyze video for cut-in and cut-out events.
    Requires lane zones to be defined and auto-label to have been run.
    """
    # Verify video exists
    video = db.query(Video).filter(Video.id == video_id).first()
    if not video:
        raise HTTPException(status_code=404, detail="Video not found")
    
    # Check if zones are defined
    zones = db.query(LaneZone).filter(LaneZone.video_id == video_id).all()
    if not zones:
        raise HTTPException(
            status_code=400, 
            detail="No lane zones defined. Please draw lane zones first."
        )
    
    # Check for ego zone
    has_ego = any(z.zone_type == "ego" for z in zones)
    if not has_ego:
        raise HTTPException(
            status_code=400,
            detail="No ego lane defined. Please draw an ego lane zone."
        )
    
    # Run analysis
    try:
        events = analyze_video_lanes(video_id, db)
        return AnalyzeResponse(
            events_detected=len(events),
            events=events
        )
    except Exception as e:
        logger.error(f"Lane analysis failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/events/{video_id}")
def clear_lane_events(video_id: int, db: Session = Depends(get_db)):
    """Clear all detected lane events for a video."""
    deleted = db.query(LaneEvent).filter(LaneEvent.video_id == video_id).delete()
    db.commit()
    
    return {"message": f"Cleared {deleted} lane events", "video_id": video_id}
