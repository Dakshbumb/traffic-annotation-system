import time
import math
import numpy as np
from sqlalchemy.orm import Session
from database import SessionLocal
import models
import crud

# Do a simple line-segment intersection check
# Line 1: p1 -> p2
# Line 2: p3 -> p4 (counting line)
def segments_intersect(p1, p2, p3, p4):
    def ccw(A, B, C):
        return (C[1]-A[1]) * (B[0]-A[0]) > (B[1]-A[1]) * (C[0]-A[0])
    
    return ccw(p1, p3, p4) != ccw(p2, p3, p4) and ccw(p1, p2, p3) != ccw(p1, p2, p4)

def run_analytics(video_id: int):
    """
    Post-process tracks for a video to compute:
    1. Line Crossings (Incoming/Outgoing counts)
    2. Speed Estimation (if calibration line exists)
    """
    db = SessionLocal()
    try:
        # Get all annotations for this video, sorted by track_id then frame
        # We need full track history to detect crossings/movement
        print(f"[analytics] Loading tracks for video {video_id}...")
        all_anns = db.query(models.Annotation).filter(
            models.Annotation.video_id == video_id,
            models.Annotation.track_id.isnot(None)
        ).order_by(models.Annotation.track_id, models.Annotation.frame_index).all()
        
        # Group by track_id
        tracks = {}
        for ann in all_anns:
            tid = ann.track_id
            if tid not in tracks:
                tracks[tid] = []
            tracks[tid].append(ann)
            
        print(f"[analytics] Processed {len(tracks)} unique tracks")

        # Get Analytics Lines
        lines = db.query(models.AnalyticsLine).filter(
            models.AnalyticsLine.video_id == video_id
        ).all()
        
        if not lines:
            print("[analytics] No analytics lines defined. Skipping.")
            return

        # Prepare counters
        line_counts = {l.id: {"in": 0, "out": 0} for l in lines}
        
        # Process each track
        for tid, track_anns in tracks.items():
            if len(track_anns) < 2:
                continue
                
            # Iterate through track history
            for i in range(len(track_anns) - 1):
                prev = track_anns[i]
                curr = track_anns[i+1]
                
                # Center points
                p1 = [(prev.x1 + prev.x2)/2, (prev.y1 + prev.y2)/2]
                p2 = [(curr.x1 + curr.x2)/2, (curr.y1 + curr.y2)/2]
                
                # Check intersection with all counting lines
                for line in lines:
                    if line.line_type == "count":
                        lp1, lp2 = line.points # [[x1,y1], [x2,y2]]
                        
                        if segments_intersect(p1, p2, lp1, lp2):
                            # Determine direction (cross product logic or simple y-diff)
                            # Simple approach: "In" if moving Down (y increasing), "Out" if moving Up
                            # Better: Vector dot product with line normal.
                            # For MVP: Let's assume Top-to-Bottom is "In", Bottom-to-Top is "Out"
                            if p2[1] > p1[1]:
                                line_counts[line.id]["in"] += 1
                            else:
                                line_counts[line.id]["out"] += 1
                                
        # Update line metadata with counts
        for line in lines:
            if line.line_type == "count":
                line.meta_data = line_counts[line.id]
                print(f"[analytics] Line '{line.name}': In={line_counts[line.id]['in']}, Out={line_counts[line.id]['out']}")
        
        db.commit()
        print("[analytics] Finished processing")
        
    except Exception as e:
        print(f"[analytics] Error: {e}")
    finally:
        db.close()

if __name__ == "__main__":
    # Test run on video 19
    run_analytics(19)
