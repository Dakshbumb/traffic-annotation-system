from typing import List
import shutil
import time

from fastapi import APIRouter, Depends, File, UploadFile, HTTPException
from sqlalchemy.orm import Session

from database import get_db
import crud
import schemas
from config import UPLOAD_DIR  # this is a pathlib.Path

router = APIRouter(
    prefix="/api/videos",
    tags=["videos"],
)


@router.post("/upload", response_model=schemas.Video)
async def upload_video(
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
):
    """
    Upload a video file and store metadata in the DB.
    Uses the SAME UPLOAD_DIR as everywhere else.
    """
    # Make sure uploads dir exists (config already does this, but harmless)
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

    # Create unique stored filename
    ts = int(time.time() * 1000)
    ext = (file.filename.rsplit(".", 1)[-1] if "." in file.filename else "mp4")
    stored_filename = f"{ts}.{ext}"

    # Destination path using pathlib
    dst_path = UPLOAD_DIR / stored_filename

    # Save file to disk
    with dst_path.open("wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Save DB record – store ONLY the name, not the full path
    video_in = schemas.VideoCreate(
        filename=stored_filename,
        original_filename=file.filename,
    )
    video = crud.create_video(db, video_in)
    return video


@router.get("/", response_model=List[schemas.Video])
def list_videos(
    skip: int = 0,
    limit: int = 50,
    db: Session = Depends(get_db),
):
    """List uploaded videos."""
    return crud.list_videos(db, skip=skip, limit=limit)


@router.get("/{video_id}", response_model=schemas.Video)
def get_video(
    video_id: int,
    db: Session = Depends(get_db),
):
    """Get single video metadata by ID."""
    video = crud.get_video(db, video_id)
    if not video:
        raise HTTPException(status_code=404, detail="Video not found")
    return video
