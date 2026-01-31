from typing import List
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

import models 
from database import get_db
import crud
import schemas

router = APIRouter(
    prefix="/api/annotations",
    tags=["annotations"],
)


# ------------------- CREATE ONE ANNOTATION -------------------
@router.post("/", response_model=schemas.Annotation)
def create_annotation(
    annotation_in: schemas.AnnotationCreate,
    db: Session = Depends(get_db),
):
    ann = crud.create_annotation(db, annotation_in)
    return ann


# ------------------- BULK CREATE ANNOTATIONS -------------------
@router.post("/bulk", response_model=dict)
def create_annotations_bulk(
    video_id: int,
    anns_in: List[schemas.AnnotationCreate],
    db: Session = Depends(get_db),
):
    if len(anns_in) == 0:
        raise HTTPException(status_code=400, detail="No annotations provided")

    crud.create_annotations_bulk(db, video_id, anns_in)
    return {"status": "ok", "inserted": len(anns_in)}


# ------------------- GET ANNOTATIONS BY VIDEO -------------------
@router.get("/video/{video_id}", response_model=List[schemas.Annotation])
def get_annotations_for_video(
    video_id: int,
    skip: int = 0,
    limit: int = 5000,
    db: Session = Depends(get_db),
):
    return crud.get_annotations_for_video(
        db, video_id=video_id, skip=skip, limit=limit
    )


# ------------------- UPDATE SINGLE ANNOTATION -------------------
@router.patch("/{annotation_id}", response_model=dict)
def update_annotation(
    annotation_id: int,
    update_data: schemas.AnnotationUpdate,
    db: Session = Depends(get_db),
):
    """Update an annotation's position, size, or class label."""
    ann = db.query(models.Annotation).filter(
        models.Annotation.id == annotation_id
    ).first()
    
    if not ann:
        raise HTTPException(status_code=404, detail="Annotation not found")
    
    # Update only provided fields
    if update_data.x1 is not None:
        ann.x1 = update_data.x1
    if update_data.y1 is not None:
        ann.y1 = update_data.y1
    if update_data.x2 is not None:
        ann.x2 = update_data.x2
    if update_data.y2 is not None:
        ann.y2 = update_data.y2
    if update_data.class_label is not None:
        ann.class_label = update_data.class_label
    if update_data.track_id is not None:
        ann.track_id = update_data.track_id
    
    db.commit()
    db.refresh(ann)
    
    return {
        "status": "updated",
        "id": annotation_id,
        "annotation": {
            "id": ann.id,
            "x1": ann.x1,
            "y1": ann.y1,
            "x2": ann.x2,
            "y2": ann.y2,
            "class_label": ann.class_label,
            "track_id": ann.track_id,
        }
    }


# ------------------- DELETE SINGLE ANNOTATION -------------------
@router.delete("/{annotation_id}", response_model=dict)
def delete_annotation(annotation_id: int, db: Session = Depends(get_db)):
    ann = db.query(crud.models.Annotation).filter(
        crud.models.Annotation.id == annotation_id
    ).first()
    if not ann:
        raise HTTPException(status_code=404, detail="Annotation not found")

    db.delete(ann)
    db.commit()
    return {"status": "deleted", "id": annotation_id}


# ------------------- DELETE ALL ANNOTATIONS FOR A VIDEO -------------------
@router.delete("/video/{video_id}", response_model=dict)
def delete_annotations_for_video(video_id: int, db: Session = Depends(get_db)):
    count = db.query(crud.models.Annotation).filter(
        crud.models.Annotation.video_id == video_id
    ).delete()
    db.commit()
    return {"status": "deleted", "count": count}


@router.get("/video/{video_id}/frame/{frame_index}")
def get_annotations_for_frame(
    video_id: int,
    frame_index: int,
    db: Session = Depends(get_db),
):
    anns = (
        db.query(models.Annotation)
        .filter(models.Annotation.video_id == video_id)
        .filter(models.Annotation.frame_index == frame_index)
        .all()
    )

    return {
        "video_id": video_id,
        "frame_index": frame_index,
        "count": len(anns),
        "annotations": [
            {
                "id": a.id,
                "track_id": a.track_id,
                "class_label": a.class_label,
                "x1": a.x1,
                "y1": a.y1,
                "x2": a.x2,
                "y2": a.y2,
            }
            for a in anns
        ],
    }
