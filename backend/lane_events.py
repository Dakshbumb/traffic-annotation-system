"""
Lane Event Detection Module

Detects cut-in and cut-out events by analyzing vehicle trajectories
relative to user-defined lane zones.

Cut-in:  Vehicle enters ego lane from adjacent lane
Cut-out: Vehicle leaves ego lane to adjacent lane
"""

from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
import logging

logger = logging.getLogger("lane_events")


@dataclass
class LaneZoneData:
    """Represents a lane zone polygon."""
    id: int
    name: str
    zone_type: str  # "ego", "adjacent_left", "adjacent_right"
    points: List[List[float]]  # [[x1,y1], [x2,y2], ...]
    color: str


@dataclass
class DetectedEvent:
    """Represents a detected lane change event."""
    track_id: int
    event_type: str  # "cut_in" or "cut_out"
    frame_index: int
    from_zone: str
    to_zone: str
    confidence: float
    bbox: List[float]  # [x1, y1, x2, y2]


def point_in_polygon(x: float, y: float, polygon: List[List[float]]) -> bool:
    """
    Check if point (x, y) is inside polygon using ray casting algorithm.
    
    Args:
        x, y: Point coordinates
        polygon: List of [x, y] vertices defining the polygon
    
    Returns:
        True if point is inside polygon
    """
    n = len(polygon)
    if n < 3:
        return False
    
    inside = False
    j = n - 1
    
    for i in range(n):
        xi, yi = polygon[i]
        xj, yj = polygon[j]
        
        if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi) + xi):
            inside = not inside
        j = i
    
    return inside


def get_bbox_center(bbox: List[float]) -> Tuple[float, float]:
    """
    Get the bottom-center point of a bounding box.
    This is typically where the vehicle contacts the ground.
    
    Args:
        bbox: [x1, y1, x2, y2]
    
    Returns:
        (x, y) bottom-center coordinates
    """
    x1, y1, x2, y2 = bbox
    return ((x1 + x2) / 2, y2)  # Bottom center


def get_zone_for_point(
    x: float, 
    y: float, 
    zones: List[LaneZoneData]
) -> Optional[str]:
    """
    Determine which zone a point belongs to.
    
    Args:
        x, y: Point coordinates
        zones: List of lane zones
    
    Returns:
        Zone type string or None if not in any zone
    """
    for zone in zones:
        if point_in_polygon(x, y, zone.points):
            return zone.zone_type
    return None


def detect_lane_events(
    tracks: Dict[int, List[Dict]],  # track_id -> list of {frame_index, bbox}
    zones: List[LaneZoneData],
    min_frames_in_zone: int = 3
) -> List[DetectedEvent]:
    """
    Detect cut-in and cut-out events from vehicle tracks.
    
    Args:
        tracks: Dictionary mapping track_id to list of detections
                Each detection: {"frame_index": int, "bbox": [x1,y1,x2,y2]}
        zones: List of lane zone definitions
        min_frames_in_zone: Minimum consecutive frames to confirm zone membership
    
    Returns:
        List of detected events
    """
    events = []
    
    # Find ego zone
    ego_zone = next((z for z in zones if z.zone_type == "ego"), None)
    if not ego_zone:
        logger.warning("No ego zone defined, cannot detect lane events")
        return events
    
    for track_id, detections in tracks.items():
        if len(detections) < 2:
            continue
        
        # Sort by frame index
        sorted_detections = sorted(detections, key=lambda d: d["frame_index"])
        
        # Track zone history
        zone_history = []
        for det in sorted_detections:
            x, y = get_bbox_center(det["bbox"])
            zone = get_zone_for_point(x, y, zones)
            zone_history.append({
                "frame": det["frame_index"],
                "zone": zone,
                "bbox": det["bbox"]
            })
        
        # Detect zone transitions
        prev_zone = None
        prev_zone_count = 0
        
        for i, entry in enumerate(zone_history):
            curr_zone = entry["zone"]
            
            if curr_zone == prev_zone:
                prev_zone_count += 1
            else:
                # Zone change detected
                if prev_zone is not None and curr_zone is not None:
                    if prev_zone_count >= min_frames_in_zone:
                        # Determine event type
                        event_type = None
                        
                        if prev_zone != "ego" and curr_zone == "ego":
                            # Vehicle entered ego lane = CUT_IN
                            event_type = "cut_in"
                        elif prev_zone == "ego" and curr_zone != "ego":
                            # Vehicle left ego lane = CUT_OUT
                            event_type = "cut_out"
                        
                        if event_type:
                            events.append(DetectedEvent(
                                track_id=track_id,
                                event_type=event_type,
                                frame_index=entry["frame"],
                                from_zone=prev_zone,
                                to_zone=curr_zone,
                                confidence=min(1.0, prev_zone_count / 10.0),
                                bbox=entry["bbox"]
                            ))
                            logger.info(
                                f"Detected {event_type}: track={track_id}, "
                                f"frame={entry['frame']}, {prev_zone} -> {curr_zone}"
                            )
                
                prev_zone = curr_zone
                prev_zone_count = 1
    
    return events


def analyze_video_lanes(
    video_id: int,
    db_session
) -> List[Dict]:
    """
    Analyze a video for lane change events using stored annotations and zones.
    
    Args:
        video_id: Database video ID
        db_session: SQLAlchemy session
    
    Returns:
        List of event dictionaries ready for database insertion
    """
    from models import Annotation, LaneZone as LaneZoneModel, LaneEvent
    
    # Load lane zones
    zone_records = db_session.query(LaneZoneModel).filter(
        LaneZoneModel.video_id == video_id
    ).all()
    
    if not zone_records:
        logger.warning(f"No lane zones defined for video {video_id}")
        return []
    
    zones = [
        LaneZoneData(
            id=z.id,
            name=z.name,
            zone_type=z.zone_type,
            points=z.points,
            color=z.color
        )
        for z in zone_records
    ]
    
    # Load vehicle tracks
    annotations = db_session.query(Annotation).filter(
        Annotation.video_id == video_id,
        Annotation.track_id.isnot(None)
    ).all()
    
    if not annotations:
        logger.warning(f"No tracked vehicles found for video {video_id}")
        return []
    
    # Group by track_id
    tracks: Dict[int, List[Dict]] = {}
    for ann in annotations:
        if ann.track_id not in tracks:
            tracks[ann.track_id] = []
        tracks[ann.track_id].append({
            "frame_index": ann.frame_index,
            "bbox": [ann.x1, ann.y1, ann.x2, ann.y2]
        })
    
    logger.info(f"Analyzing {len(tracks)} tracks across {len(zones)} zones")
    
    # Detect events
    events = detect_lane_events(tracks, zones)
    
    # Clear existing events for this video
    db_session.query(LaneEvent).filter(LaneEvent.video_id == video_id).delete()
    
    # Store new events
    event_dicts = []
    for event in events:
        lane_event = LaneEvent(
            video_id=video_id,
            track_id=event.track_id,
            event_type=event.event_type,
            frame_index=event.frame_index,
            from_zone=event.from_zone,
            to_zone=event.to_zone,
            confidence=event.confidence,
            bbox=event.bbox
        )
        db_session.add(lane_event)
        event_dicts.append({
            "track_id": event.track_id,
            "event_type": event.event_type,
            "frame_index": event.frame_index,
            "from_zone": event.from_zone,
            "to_zone": event.to_zone,
            "confidence": event.confidence,
            "bbox": event.bbox
        })
    
    db_session.commit()
    logger.info(f"Stored {len(event_dicts)} lane events for video {video_id}")
    
    return event_dicts
