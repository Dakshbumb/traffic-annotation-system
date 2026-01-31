from typing import Optional
from uuid import uuid4
from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from pydantic import BaseModel

from database import get_db
import crud
from job_store import save_job, get_job, job_store


router = APIRouter(
    prefix="/api/autolabel",
    tags=["autolabel"],
)


class AutolabelRequest(BaseModel):
    video_id: int
    model: str = "yolov8n"
    confidence_threshold: float = 0.4
    frame_stride: int = 1
    start_frame: int = 0
    end_frame: Optional[int] = None
    overwrite: bool = True


@router.post("/video")
def start_autolabel(
    req: AutolabelRequest,
    db: Session = Depends(get_db),
):
    """
    Create a new autolabel job for a given video.
    For now, we only queue the job in memory (job_store).
    """
    # 1) Make sure the video exists
    video = crud.get_video(db, req.video_id)
    if not video:
        raise HTTPException(status_code=404, detail="Video not found")

    # 2) Generate a job id
    job_id = f"alb_{uuid4().hex[:12]}"

    now = datetime.utcnow().isoformat()

    # 3) Build job data
    job_data = {
        "job_id": job_id,
        "video_id": req.video_id,
        "status": "queued",          # queued / running / completed / failed
        "progress": 0.0,             # 0.0 .. 1.0
        "created_at": now,
        "updated_at": now,
        "params": req.model_dump(),
        "error_message": None,
    }

    # 4) Save into in-memory store
    save_job(job_id, job_data)

    # 5) Return to client
    return job_data


@router.get("/jobs/{job_id}")
def get_autolabel_job(job_id: str):
    """
    Get status for a specific autolabel job.
    """
    job = get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return job


@router.get("/jobs")
def list_autolabel_jobs(video_id: Optional[int] = None):
    """
    List all jobs in memory. Optional filter by video_id.
    """
    jobs = list(job_store.values())
    if video_id is not None:
        jobs = [j for j in jobs if j["video_id"] == video_id]
    return {"jobs": jobs}
