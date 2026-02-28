from typing import List
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
    # --- Validation ---
    ALLOWED_EXTENSIONS = {"mp4", "avi", "mov", "mkv", "webm"}
    MAX_FILE_SIZE = 2 * 1024 * 1024 * 1024  # 2 GB

    ext = (file.filename.rsplit(".", 1)[-1].lower() if "." in file.filename else "")
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type '.{ext}'. Allowed: {', '.join(sorted(ALLOWED_EXTENSIONS))}",
        )

    # Make sure uploads dir exists
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

    # Create unique stored filename
    ts = int(time.time() * 1000)
    stored_filename = f"{ts}.{ext}"

    # Destination path using pathlib
    dst_path = UPLOAD_DIR / stored_filename

    # Save file to disk using async chunked writes (1 MB at a time)
    # Also enforce max file size during upload
    CHUNK_SIZE = 1024 * 1024  # 1 MB
    total_written = 0
    with dst_path.open("wb") as buffer:
        while True:
            chunk = await file.read(CHUNK_SIZE)
            if not chunk:
                break
            total_written += len(chunk)
            if total_written > MAX_FILE_SIZE:
                buffer.close()
                dst_path.unlink(missing_ok=True)
                raise HTTPException(
                    status_code=413,
                    detail=f"File too large. Maximum size is {MAX_FILE_SIZE // (1024*1024*1024)} GB.",
                )
            buffer.write(chunk)

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
