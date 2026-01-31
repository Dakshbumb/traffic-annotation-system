from sqlalchemy.orm import Session
from collections import defaultdict
import models
import schemas
from analytics_utils import do_intersect

def process_video_analytics(db: Session, video_id: int):
    """
    Recalculate counts for all lines in the video based on existing annotations.
    """
    # 1. Fetch all analytics lines
    lines = db.query(models.AnalyticsLine).filter(models.AnalyticsLine.video_id == video_id).all()
    if not lines:
        print(f"[Analytics] No lines found for video {video_id}")
        return

    # Reset counts
    line_counts = {line.id: {"in": 0, "out": 0} for line in lines}

    # 2. Fetch all annotations for the video
    # We only care about objects with track_ids
    annotations = db.query(models.Annotation).filter(
        models.Annotation.video_id == video_id,
        models.Annotation.track_id.isnot(None)
    ).order_by(models.Annotation.frame_index).all()

    if not annotations:
        print(f"[Analytics] No annotations found for video {video_id}")
        return

    # 3. Group by track_id to build trajectories
    # track_id -> list of (x_center, y_center) sorted by frame
    tracks = defaultdict(list)
    
    for ann in annotations:
        # Calculate center bottom point (usually better for "crossing" logic on roads)
        # or just center. Let's use center.
        cx = (ann.x1 + ann.x2) / 2
        cy = (ann.y1 + ann.y2) / 2
        tracks[ann.track_id].append((cx, cy))

    # 4. Check intersections
    for track_id, points in tracks.items():
        if len(points) < 2:
            continue

        # Iterate through segments of the track trajectory
        for i in range(len(points) - 1):
            p1 = points[i]
            p2 = points[i+1]

            # Check against all lines
            for line in lines:
                # line.points is [[x1, y1], [x2, y2]]
                # Validation
                if not line.points or len(line.points) != 2:
                    continue
                
                l_p1 = line.points[0]
                l_p2 = line.points[1]

                if do_intersect(p1, p2, l_p1, l_p2):
                    # Determine direction
                    # We can use Cross Product to determine "In" vs "Out"
                    # Vector of line: V_line = l_p2 - l_p1
                    # Vector of movement: V_move = p2 - p1
                    # signal = (l_x2 - l_x1) * (p_y2 - p_y1) - (l_y2 - l_y1) * (p_x2 - p_x1)
                    
                    # Simplified: Just check orientation relative to the line vector
                    
                    dx_line = l_p2[0] - l_p1[0]
                    dy_line = l_p2[1] - l_p1[1]
                    
                    dx_move = p2[0] - p1[0]
                    dy_move = p2[1] - p1[1]

                    # 2D cross product of Line Vector and Movement Vector?
                    # Or Cross product of Line Vector and (P1 - LineStart) vs (P2 - LineStart)?
                    # Let's define "In" as crossing from Right-to-Left relative to line direction?
                    # Or just arbitrary for now.
                    
                    # Let's use the cross product of the line vector and the trajectory vector
                    # If positive -> 'In', negative -> 'Out' (arbitrary convention)
                    cross_prod = dx_line * dy_move - dy_line * dx_move
                    
                    if cross_prod > 0:
                        line_counts[line.id]["in"] += 1
                    else:
                        line_counts[line.id]["out"] += 1
                    
                    # Optimization: A track should usually only count once per line? 
                    # But it might cross back and forth. 
                    # For traffic counting, usually count once.
                    # But for now, let's count every crossing.
                    
    # 5. Update DB
    for line in lines:
        line.meta_data = line_counts[line.id] 
        # Ensure we construct a new dict/json to force SQLAlchemy update if needed (mutable dict issue)
        # But assigning a new dict works.
    
    db.commit()
    print(f"[Analytics] Processed video {video_id}. Counts: {line_counts}")

