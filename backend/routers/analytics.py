from typing import List
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from database import get_db
import models
import schemas
from analytics_processor import process_video_analytics

router = APIRouter(
    prefix="/api/analytics",
    tags=["analytics"],
)

# ------------------- CREATE LINE -------------------
@router.post("/", response_model=schemas.AnalyticsLine)
def create_analytics_line(
    line: schemas.AnalyticsLineCreate,
    db: Session = Depends(get_db),
):
    # Check if video exists
    video = db.query(models.Video).filter(models.Video.id == line.video_id).first()
    if not video:
        raise HTTPException(status_code=404, detail="Video not found")

    db_line = models.AnalyticsLine(
        video_id=line.video_id,
        name=line.name,
        line_type=line.line_type,
        points=line.points,
        meta_data=line.meta_data,
    )
    db.add(db_line)
    db.commit()
    db.refresh(db_line)
    return db_line

# ------------------- LIST LINES -------------------
@router.get("/video/{video_id}", response_model=List[schemas.AnalyticsLine])
def list_analytics_lines(
    video_id: int,
    db: Session = Depends(get_db),
):
    lines = db.query(models.AnalyticsLine).filter(
        models.AnalyticsLine.video_id == video_id
    ).all()
    return lines

# ------------------- DELETE LINE -------------------
@router.delete("/{line_id}")
def delete_analytics_line(
    line_id: int,
    db: Session = Depends(get_db),
):
    line = db.query(models.AnalyticsLine).filter(models.AnalyticsLine.id == line_id).first()
    if not line:
        raise HTTPException(status_code=404, detail="Line not found")
    
    db.delete(line)
    db.commit()
    return {"status": "deleted", "id": line_id}

# ------------------- RUN ANALYTICS -------------------
@router.post("/process/{video_id}")
def run_analytics_process(
    video_id: int,
    db: Session = Depends(get_db),
):
    """
    Trigger the analytics processing (line crossing counting) for a video.
    This runs synchronously for now as it's fast.
    """
    video = db.query(models.Video).filter(models.Video.id == video_id).first()
    if not video:
        raise HTTPException(status_code=404, detail="Video not found")
        
    try:
        process_video_analytics(db, video_id)
        return {"status": "completed", "video_id": video_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
