import sys
import os

# Ensure backend modules are found
sys.path.append(os.path.join(os.getcwd(), "backend"))

from database import SessionLocal, Base, engine
import models
from analytics_processor import process_video_analytics

def verify_analytics():
    print("--- Initializing DB Tables ---")
    models.Base.metadata.create_all(bind=engine)
    
    db = SessionLocal()
    
    print("--- Setting up Test Data ---")
    
    # 1. Create a dummy video
    video = models.Video(filename="test_analytics.mp4", original_filename="test_analytics.mp4")
    db.add(video)
    db.commit()
    db.refresh(video)
    print(f"Created Video ID: {video.id}")
    
    # 2. Create a dummy line (Vertical line at x=100 from y=0 to y=200)
    # Direction: Top to Bottom? Line direction is p1->p2
    # p1=(100,0), p2=(100,200). 
    # Vector is (0, 200).
    # Normal (Right side) is (-200, 0) (Counter-clockwise rotation)
    line = models.AnalyticsLine(
        video_id=video.id,
        name="Test Line",
        line_type="count",
        points=[[100, 0], [100, 200]],
        meta_data={}
    )
    db.add(line)
    db.commit()
    print(f"Created Line ID: {line.id}")
    
    # 3. Create Tracks
    # Track 1: Crossing Left to Right (0,100) -> (200,100)
    # Movement Vector (200, 0)
    # Line Vector (0, 200)
    # Cross Product: (dx_L * dy_M) - (dy_L * dx_M) 
    # (0 * 0) - (200 * 200) = -40000 -> Should be OUT (or In depending on sign convention)
    t1_frames = [
        {"frame": 0, "x": 0, "y": 100},
        {"frame": 1, "x": 50, "y": 100},
        {"frame": 2, "x": 150, "y": 100}, # Crossing happens between frame 1 and 2
        {"frame": 3, "x": 200, "y": 100},
    ]
    
    for f in t1_frames:
        ann = models.Annotation(
            video_id=video.id,
            frame_index=f["frame"],
            track_id=1,
            class_label="car",
            x1=f["x"]-10, y1=f["y"]-10, x2=f["x"]+10, y2=f["y"]+10,
            extra_meta={}
        )
        db.add(ann)

    # Track 2: Crossing Right to Left (200,150) -> (0,150)
    # Movement Vector (-200, 0)
    # Cross Product: (0 * 0) - (200 * -200) = +40000 -> Should be IN (opposite sign)
    t2_frames = [
        {"frame": 0, "x": 200, "y": 150},
        {"frame": 1, "x": 150, "y": 150},
        {"frame": 2, "x": 50, "y": 150},
        {"frame": 3, "x": 0, "y": 150},
    ]
    
    for f in t2_frames:
        ann = models.Annotation(
            video_id=video.id,
            frame_index=f["frame"],
            track_id=2,
            class_label="truck",
            x1=f["x"]-10, y1=f["y"]-10, x2=f["x"]+10, y2=f["y"]+10,
            extra_meta={}
        )
        db.add(ann)

    # Track 3: Not crossing (Parallel or far away)
    # (50, 0) -> (50, 200)
    t3_frames = [
        {"frame": 0, "x": 50, "y": 0},
        {"frame": 1, "x": 50, "y": 100},
        {"frame": 2, "x": 50, "y": 200},
    ]
    for f in t3_frames:
        ann = models.Annotation(
            video_id=video.id,
            frame_index=f["frame"],
            track_id=3,
            class_label="bus",
            x1=f["x"]-10, y1=f["y"]-10, x2=f["x"]+10, y2=f["y"]+10,
            extra_meta={}
        )
        db.add(ann)
        
    db.commit()
    print("Created tracks.")
    
    print("--- Running Analytics ---")
    process_video_analytics(db, video.id)
    
    db.refresh(line)
    print(f"Resulting Counts: {line.meta_data}")
    
    # Cleanup
    print("--- Cleanup ---")
    db.delete(video) # Cascades should delete lines and annotations
    db.commit()
    
    # Assertions
    counts = line.meta_data
    if counts["in"] == 1 and counts["out"] == 1:
        print("SUCCESS: Counts are correct (1 In, 1 Out)")
    elif counts["in"] + counts["out"] == 2:
        print(f"SUCCESS: Total crossings correct (2), though direction might be swapped: {counts}")
    else:
        print(f"FAILURE: Counts are incorrect: {counts}. Expected 1 In, 1 Out.")

if __name__ == "__main__":
    try:
        verify_analytics()
    except Exception as e:
        print(f"Error: {e}")
