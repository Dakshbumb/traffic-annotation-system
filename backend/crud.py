from typing import List, Optional

from sqlalchemy.orm import Session

import models
import schemas


# ---------- VIDEO CRUD ----------

def create_video(db: Session, video_in: schemas.VideoCreate) -> models.Video:
    video = models.Video(
        filename=video_in.filename,
        original_filename=video_in.original_filename,
    )
    db.add(video)
    db.commit()
    db.refresh(video)
    return video


def get_video(db: Session, video_id: int) -> Optional[models.Video]:
    return db.query(models.Video).filter(models.Video.id == video_id).first()


def list_videos(db: Session, skip: int = 0, limit: int = 50) -> List[models.Video]:
    return (
        db.query(models.Video)
        .order_by(models.Video.created_at.desc())
        .offset(skip)
        .limit(limit)
        .all()
    )


# ---------- ANNOTATION CRUD ----------

def create_annotation(db: Session, annotation_in: schemas.AnnotationCreate) -> models.Annotation:
    annotation = models.Annotation(
        video_id=annotation_in.video_id,
        frame_index=annotation_in.frame_index,
        track_id=annotation_in.track_id,
        class_label=annotation_in.class_label,
        x1=annotation_in.x1,
        y1=annotation_in.y1,
        x2=annotation_in.x2,
        y2=annotation_in.y2,
        extra_meta=annotation_in.extra_meta,
    )
    db.add(annotation)
    db.commit()
    db.refresh(annotation)
    return annotation


def create_annotations_bulk(
    db: Session,
    video_id: int,
    anns_in: List[schemas.AnnotationCreate],
) -> List[models.Annotation]:
    result: List[models.Annotation] = []

    for ann_in in anns_in:
        annotation = models.Annotation(
            video_id=video_id,
            frame_index=ann_in.frame_index,
            track_id=ann_in.track_id,
            class_label=ann_in.class_label,
            x1=ann_in.x1,
            y1=ann_in.y1,
            x2=ann_in.x2,
            y2=ann_in.y2,
            extra_meta=ann_in.extra_meta,
        )
        db.add(annotation)
        result.append(annotation)

    db.commit()

    for a in result:
        db.refresh(a)

    return result


def get_annotations_for_video(
    db: Session,
    video_id: int,
    skip: int = 0,
    limit: int = 5000,
) -> List[models.Annotation]:
    return (
        db.query(models.Annotation)
        .filter(models.Annotation.video_id == video_id)
        .order_by(models.Annotation.frame_index.asc())
        .offset(skip)
        .limit(limit)
        .all()
    )

def get_track_annotations_in_range(db, video_id, track_id, start_frame, end_frame):
    return (
        db.query(models.Annotation)
        .filter(models.Annotation.video_id == video_id)
        .filter(models.Annotation.track_id == track_id)
        .filter(models.Annotation.frame_index >= start_frame)
        .filter(models.Annotation.frame_index <= end_frame)
        .order_by(models.Annotation.frame_index.asc())
        .all()
    )

def find_track_frame(db, video_id, track_id, frame_index):
    return (
        db.query(models.Annotation)
        .filter(models.Annotation.video_id == video_id)
        .filter(models.Annotation.track_id == track_id)
        .filter(models.Annotation.frame_index == frame_index)
        .first()
    )


import uuid
from datetime import datetime

def create_autolabel_job(db: Session, video_id: int, params: dict) -> models.AutolabelJob:
    job_uuid = f"alb_{uuid.uuid4().hex[:16]}"
    job = models.AutolabelJob(
        job_id=job_uuid,
        video_id=video_id,
        status="queued",
        progress=0.0,
        params=params,
    )
    db.add(job)
    db.commit()
    db.refresh(job)
    return job


def get_autolabel_job_by_job_id(db: Session, job_id: str) -> Optional[models.AutolabelJob]:
    return (
        db.query(models.AutolabelJob)
        .filter(models.AutolabelJob.job_id == job_id)
        .first()
    )


def list_autolabel_jobs_for_video(db: Session, video_id: int) -> List[models.AutolabelJob]:
    return (
        db.query(models.AutolabelJob)
        .filter(models.AutolabelJob.video_id == video_id)
        .order_by(models.AutolabelJob.created_at.desc())
        .all()
    )


def update_autolabel_job_status(
    db: Session,
    job: models.AutolabelJob,
    status: str,
    progress: float | None = None,
    error_message: str | None = None,
) -> models.AutolabelJob:
    job.status = status
    if progress is not None:
        job.progress = progress
    if error_message is not None:
        job.error_message = error_message
    job.updated_at = datetime.utcnow()
    db.add(job)
    db.commit()
    db.refresh(job)
    return job

def get_annotations_for_frame(
    db: Session,
    video_id: int,
    frame_index: int,
):
    return (
        db.query(models.Annotation)
        .filter(
            models.Annotation.video_id == video_id,
            models.Annotation.frame_index == frame_index,
        )
        .all()
    )


