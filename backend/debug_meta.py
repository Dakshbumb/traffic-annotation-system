from sqlalchemy.orm import Session
from database import SessionLocal
import models
import json

db = SessionLocal()
ann = db.query(models.Annotation).filter(models.Annotation.video_id == 19).first()
if ann:
    print(f"ID: {ann.id}")
    print(f"Class: {ann.class_label}")
    print(f"Extra Meta: {ann.extra_meta}")
    print(f"Type: {type(ann.extra_meta)}")
else:
    print("No annotations found")
db.close()
