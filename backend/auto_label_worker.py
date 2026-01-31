import time
from typing import Optional

from sqlalchemy.orm import Session

from database import SessionLocal
import crud
import models


POLL_INTERVAL_SECONDS = 3  # how often worker checks for new jobs


def get_next_queued_job(db: Session) -> Optional[models.AutolabelJob]:
    return (
        db.query(models.AutolabelJob)
        .filter(models.AutolabelJob.status == "queued")
        .order_by(models.AutolabelJob.created_at.asc())
        .first()
    )


def process_job(db: Session, job: models.AutolabelJob):
    print(f"[worker] Starting job {job.job_id} for video {job.video_id}")
    crud.update_autolabel_job_status(db, job, status="running", progress=0.0)

    try:
        # -----------------------------
        # TODO: REPLACE THIS WITH REAL
        # YOLO + DeepSORT PIPELINE
        # -----------------------------
        #
        # 1. open video file based on job.video_id
        # 2. run object detector on frames
        # 3. run DeepSORT tracker
        # 4. build list[schemas.AnnotationCreate]
        # 5. call crud.create_annotations_bulk(...)
        #
        # For now we just sleep & fake progress
        steps = 5
        for i in range(steps):
            time.sleep(1)  # simulate work
            progress = (i + 1) / steps
            crud.update_autolabel_job_status(db, job, status="running", progress=progress)
            print(f"[worker] Job {job.job_id} progress: {progress*100:.0f}%")

        # Mark as completed
        crud.update_autolabel_job_status(db, job, status="completed", progress=1.0)
        print(f"[worker] Job {job.job_id} completed")

    except Exception as e:
        print(f"[worker] Job {job.job_id} failed: {e}")
        crud.update_autolabel_job_status(
            db,
            job,
            status="failed",
            error_message=str(e),
        )


def main_loop():
    print("[worker] Autolabel worker started")
    while True:
        db: Session = SessionLocal()
        try:
            job = get_next_queued_job(db)
            if job:
                process_job(db, job)
            else:
                # nothing to do, sleep
                time.sleep(POLL_INTERVAL_SECONDS)
        finally:
            db.close()


if __name__ == "__main__":
    main_loop()
