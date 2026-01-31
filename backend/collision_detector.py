"""
Collision Detector Module

Detects near-miss events and potential collisions by calculating 
Time-To-Collision (TTC) between tracked vehicles.
"""

import math
import logging
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass

logger = logging.getLogger("collision_detector")

# Thresholds
TTC_CRITICAL_THRESHOLD = 1.5  # Seconds
TTC_WARNING_THRESHOLD = 3.0   # Seconds
PROXIMITY_THRESHOLD = 50.0    # Pixels (minimum distance to consider)


@dataclass
class VehicleState:
    """State of a vehicle at a specific frame."""
    frame_index: int
    track_id: int
    bbox: List[float]  # [x1, y1, x2, y2]
    timestamp: float   # Seconds
    
    @property
    def center(self) -> Tuple[float, float]:
        """Get center (x, y) of the bounding box."""
        return ((self.bbox[0] + self.bbox[2]) / 2, (self.bbox[1] + self.bbox[3]) / 2)


@dataclass
class DetectedNearMiss:
    """Represents a detected near-miss event."""
    frame_index: int
    track_id_1: int
    track_id_2: int
    ttc: float
    distance: float
    relative_speed: float
    severity: str  # "warning", "critical"


def calculate_distance(p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
    """Calculate Euclidean distance between two points."""
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)


def calculate_ttc(
    v1_curr: VehicleState, v1_prev: VehicleState,
    v2_curr: VehicleState, v2_prev: VehicleState,
    fps: float
) -> Optional[float]:
    """
    Calculate Time-To-Collision (TTC) between two vehicles.
    
    TTC = Distance / Relative_Velocity
    Only considers closing velocity (vehicles moving closer).
    """
    # Current positions
    c1 = v1_curr.center
    c2 = v2_curr.center
    dist_curr = calculate_distance(c1, c2)
    
    if dist_curr < 1e-3:
        return 0.0  # Already collided
        
    # Previous positions
    p1 = v1_prev.center
    p2 = v2_prev.center
    dist_prev = calculate_distance(p1, p2)
    
    # Calculate closing speed (how much distance reduced per second)
    # dist_diff = dist_prev - dist_curr
    # If positive, they are getting closer.
    
    dist_change_per_frame = dist_prev - dist_curr
    closing_speed = dist_change_per_frame * fps
    
    if closing_speed <= 0:
        return None  # Moving away or constant distance
        
    ttc = dist_curr / closing_speed
    return ttc


def detect_near_misses(
    tracks: Dict[int, List[Dict]],  # track_id -> list of detection dicts
    fps: float = 30.0
) -> List[DetectedNearMiss]:
    """
    Detect near-miss events from vehicle tracks.
    
    Args:
        tracks: Dictionary of vehicle tracks
        fps: Video frame rate
        
    Returns:
        List of detected near-miss events
    """
    events = []
    
    # Get all active track IDs
    track_ids = list(tracks.keys())
    
    # Build frame-indexed lookup
    # frame_index -> {track_id: VehicleState}
    frames_data: Dict[int, Dict[int, VehicleState]] = {}
    
    max_frame = 0
    for tid, detections in tracks.items():
        for det in detections:
            f_idx = det["frame_index"]
            max_frame = max(max_frame, f_idx)
            if f_idx not in frames_data:
                frames_data[f_idx] = {}
            
            frames_data[f_idx][tid] = VehicleState(
                frame_index=f_idx,
                track_id=tid,
                bbox=det["bbox"],
                timestamp=f_idx / fps
            )
            
    # Iterate through frames to check interactions
    # We need history, so start from frame 1 (or min frame + 1)
    
    sorted_frames = sorted(frames_data.keys())
    if not sorted_frames:
        return []
        
    for i in range(1, len(sorted_frames)):
        curr_f_idx = sorted_frames[i]
        prev_f_idx = sorted_frames[i-1]
        
        # If frames are not consecutive, skip (gap in data)
        # Allow small gap? For now strict consecutive or small skip
        if curr_f_idx - prev_f_idx > 3:
            continue
            
        current_vehicles = frames_data[curr_f_idx]
        prev_vehicles = frames_data[prev_f_idx]
        
        # Check all pairs in current frame
        active_ids = list(current_vehicles.keys())
        
        for idx1 in range(len(active_ids)):
            for idx2 in range(idx1 + 1, len(active_ids)):
                tid1 = active_ids[idx1]
                tid2 = active_ids[idx2]
                
                # Must exist in previous frame too for velocity calc
                if tid1 not in prev_vehicles or tid2 not in prev_vehicles:
                    continue
                    
                v1_curr = current_vehicles[tid1]
                v2_curr = current_vehicles[tid2]
                v1_prev = prev_vehicles[tid1]
                v2_prev = prev_vehicles[tid2]
                
                # Check proximity first
                dist = calculate_distance(v1_curr.center, v2_curr.center)
                if dist > 200: # Optimization: Don't calculate TTC if far apart
                    continue
                    
                ttc = calculate_ttc(v1_curr, v1_prev, v2_curr, v2_prev, fps)
                
                if ttc is not None and ttc < TTC_WARNING_THRESHOLD:
                    severity = "critical" if ttc < TTC_CRITICAL_THRESHOLD else "warning"
                    
                    # Store event
                    # Optimization: Don't spam events. Only log local minima of TTC? 
                    # For now log all, frontend/API can filter or we filter later.
                    # Or simpler: Just log it.
                    
                    events.append(DetectedNearMiss(
                        frame_index=curr_f_idx,
                        track_id_1=tid1,
                        track_id_2=tid2,
                        ttc=round(ttc, 2),
                        distance=round(dist, 1),
                        relative_speed=round(dist / ttc, 1) if ttc > 0 else 0, # approx closing speed
                        severity=severity
                    ))
                    
    # Filter event stream to reduce noise?
    # e.g., group consecutive events for same pair and pick worst TTC
    consolidated_events = consolidate_events(events)
    
    return consolidated_events


def consolidate_events(raw_events: List[DetectedNearMiss]) -> List[DetectedNearMiss]:
    """
    Group consecutive detections of the same event and return the peak (min TTC).
    """
    if not raw_events:
        return []
        
    # Group by pair (tid1, tid2) - normalized so min id first
    groups = {}
    
    for evt in raw_events:
        pair = tuple(sorted((evt.track_id_1, evt.track_id_2)))
        if pair not in groups:
            groups[pair] = []
        groups[pair].append(evt)
        
    final_events = []
    
    for pair, group in groups.items():
        # Sort by frame
        group.sort(key=lambda x: x.frame_index)
        
        # Split into clusters (if gap > 30 frames / 1 second, it's a new event)
        clusters = []
        current_cluster = [group[0]]
        
        for i in range(1, len(group)):
            if group[i].frame_index - group[i-1].frame_index < 30:
                current_cluster.append(group[i])
            else:
                clusters.append(current_cluster)
                current_cluster = [group[i]]
        clusters.append(current_cluster)
        
        # For each cluster, find the one with min TTC
        for cluster in clusters:
            if not cluster:
                continue
            best_event = min(cluster, key=lambda x: x.ttc)
            final_events.append(best_event)
            
    return sorted(final_events, key=lambda x: x.frame_index)

