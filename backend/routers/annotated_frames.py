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

@router.get("/video/{video_id}/{frame_index}/annotated")
def get_annotated_frame(
    video_id: int,
    frame_index: int,
    db: Session = Depends(get_db),
):
    # 1. Get video
    video = crud.get_video(db, video_id)
    if not video:
        raise HTTPException(status_code=404, detail="Video not found")

    video_path = os.path.join(UPLOAD_DIR, video.filename)
    if not os.path.exists(video_path):
        raise HTTPException(status_code=404, detail="Video file missing")

    # 2. Read frame
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ret, frame = cap.read()
    cap.release()

    if not ret:
        raise HTTPException(status_code=404, detail="Frame not found")

    # 3. Fetch annotations for this frame
    annotations = crud.get_annotations_for_frame(
        db=db,
        video_id=video_id,
        frame_index=frame_index,
    )

    # 4. Draw boxes
    for ann in annotations:
        x1, y1, x2, y2 = map(int, [ann.x1, ann.y1, ann.x2, ann.y2])

        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2,
        )

        label = ann.class_label
        if ann.track_id is not None and ann.track_id >= 0:
            label += f" #{ann.track_id}"

        cv2.putText(
            frame,
            label,
            (x1, max(y1 - 5, 15)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1,
            cv2.LINE_AA,
        )

    # 5. Encode JPEG
    ok, jpeg = cv2.imencode(".jpg", frame)
    if not ok:
        raise HTTPException(status_code=500, detail="Failed to encode image")

    return Response(
        content=jpeg.tobytes(),
        media_type="image/jpeg",
    )
