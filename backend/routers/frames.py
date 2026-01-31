import cv2
import os
from fastapi import APIRouter, HTTPException, Depends
from fastapi.responses import Response
from sqlalchemy.orm import Session

from config import UPLOAD_DIR
from database import get_db
import crud

router = APIRouter(
    prefix="/api/frames",
    tags=["frames"],
)

@router.get("/video/{video_id}/{frame_index}")
def get_video_frame(
    video_id: int,
    frame_index: int,
    db: Session = Depends(get_db),
):
    video = crud.get_video(db, video_id)
    if not video:
        raise HTTPException(status_code=404, detail="Video not found")

    video_path = os.path.join(UPLOAD_DIR, video.filename)
    if not os.path.exists(video_path):
        raise HTTPException(status_code=404, detail="Video file not found")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise HTTPException(status_code=500, detail="Failed to open video")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if frame_index < 0 or frame_index >= total_frames:
        cap.release()
        raise HTTPException(status_code=400, detail="Frame index out of range")

    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ret, frame = cap.read()
    cap.release()

    if not ret:
        raise HTTPException(status_code=500, detail="Failed to read frame")

    success, jpeg = cv2.imencode(".jpg", frame)
    if not success:
        raise HTTPException(status_code=500, detail="Failed to encode frame")

    return Response(
        content=jpeg.tobytes(),
        media_type="image/jpeg"
    )
