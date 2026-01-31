from sqlalchemy.orm import Session
from database import SessionLocal
import models
import json

db = SessionLocal()
# Count total vs count with confidence
total = db.query(models.Annotation).filter(models.Annotation.video_id == 19).count()
valid = 0
anns = db.query(models.Annotation).filter(models.Annotation.video_id == 19).all()

for ann in anns:
    if ann.extra_meta and 'confidence' in ann.extra_meta:
        valid += 1
    else:
        pass
        # print(f"Invalid: {ann.id} Meta: {ann.extra_meta}")

print(f"Total: {total}")
print(f"Valid (has confidence): {valid}")
db.close()
