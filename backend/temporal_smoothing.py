"""
Temporal Smoothing for Bounding Boxes
Reduces jitter in tracked boxes across frames using exponential moving average.
"""

from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from collections import defaultdict


@dataclass
class SmoothBox:
    """Smoothed bounding box state."""
    x1: float
    y1: float
    x2: float
    y2: float
    
    def to_tuple(self) -> Tuple[float, float, float, float]:
        return (self.x1, self.y1, self.x2, self.y2)


class TemporalSmoother:
    """
    Applies exponential moving average smoothing to track bounding boxes.
    
    This reduces jitter/noise in bounding box positions across frames,
    making annotations appear more stable and consistent.
    
    Usage:
        smoother = TemporalSmoother(alpha=0.7)
        smoothed_box = smoother.smooth(track_id, x1, y1, x2, y2)
    """
    
    def __init__(self, alpha: float = 0.7):
        """
        Initialize smoother.
        
        Args:
            alpha: Smoothing factor (0-1). Higher = more responsive, lower = more smooth.
                   Default 0.7 provides good balance between smoothness and responsiveness.
        """
        self.alpha = alpha
        self._track_states: Dict[int, SmoothBox] = {}
    
    def smooth(
        self,
        track_id: int,
        x1: float,
        y1: float,
        x2: float,
        y2: float,
    ) -> Tuple[float, float, float, float]:
        """
        Smooth a bounding box using EMA.
        
        Args:
            track_id: Unique track identifier
            x1, y1, x2, y2: Current bounding box coordinates
        
        Returns:
            Tuple of smoothed (x1, y1, x2, y2)
        """
        if track_id not in self._track_states:
            # First observation - no smoothing needed
            self._track_states[track_id] = SmoothBox(x1, y1, x2, y2)
            return (x1, y1, x2, y2)
        
        prev = self._track_states[track_id]
        
        # Exponential moving average
        smoothed = SmoothBox(
            x1=self.alpha * x1 + (1 - self.alpha) * prev.x1,
            y1=self.alpha * y1 + (1 - self.alpha) * prev.y1,
            x2=self.alpha * x2 + (1 - self.alpha) * prev.x2,
            y2=self.alpha * y2 + (1 - self.alpha) * prev.y2,
        )
        
        self._track_states[track_id] = smoothed
        return smoothed.to_tuple()
    
    def remove_track(self, track_id: int):
        """Remove a track from smoothing state."""
        if track_id in self._track_states:
            del self._track_states[track_id]
    
    def clear(self):
        """Clear all track states."""
        self._track_states.clear()
    
    def get_track_count(self) -> int:
        """Get number of active tracks being smoothed."""
        return len(self._track_states)


class AdaptiveSmoother(TemporalSmoother):
    """
    Adaptive smoother that adjusts alpha based on motion.
    
    When object is moving fast, uses higher alpha (less smoothing).
    When object is stationary, uses lower alpha (more smoothing).
    """
    
    def __init__(
        self,
        base_alpha: float = 0.7,
        min_alpha: float = 0.3,
        max_alpha: float = 0.9,
        motion_threshold: float = 20.0,
    ):
        super().__init__(alpha=base_alpha)
        self._prev_centers: Dict[int, Tuple[float, float]] = {}
        self.base_alpha = base_alpha
        self.min_alpha = min_alpha
        self.max_alpha = max_alpha
        self.motion_threshold = motion_threshold
    
    def smooth(
        self,
        track_id: int,
        x1: float,
        y1: float,
        x2: float,
        y2: float,
    ) -> Tuple[float, float, float, float]:
        """Smooth with adaptive alpha based on motion."""
        # Calculate center
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        
        # Calculate motion
        if track_id in self._prev_centers:
            prev_cx, prev_cy = self._prev_centers[track_id]
            motion = ((cx - prev_cx) ** 2 + (cy - prev_cy) ** 2) ** 0.5
            
            # Adapt alpha based on motion
            motion_ratio = min(motion / self.motion_threshold, 1.0)
            self.alpha = self.min_alpha + motion_ratio * (self.max_alpha - self.min_alpha)
        
        self._prev_centers[track_id] = (cx, cy)
        
        return super().smooth(track_id, x1, y1, x2, y2)
    
    def remove_track(self, track_id: int):
        super().remove_track(track_id)
        if track_id in self._prev_centers:
            del self._prev_centers[track_id]
    
    def clear(self):
        super().clear()
        self._prev_centers.clear()


def smooth_annotations_batch(
    annotations: List[dict],
    alpha: float = 0.7,
) -> List[dict]:
    """
    Apply temporal smoothing to a batch of annotations.
    
    Args:
        annotations: List of annotation dicts with track_id, x1, y1, x2, y2
        alpha: Smoothing factor
    
    Returns:
        List of annotations with smoothed coordinates
    """
    smoother = TemporalSmoother(alpha=alpha)
    
    # Sort by frame to ensure temporal order
    sorted_anns = sorted(annotations, key=lambda a: a.get("frame_index", 0))
    
    result = []
    for ann in sorted_anns:
        track_id = ann.get("track_id")
        if track_id is None:
            result.append(ann)
            continue
        
        x1, y1, x2, y2 = smoother.smooth(
            track_id,
            ann["x1"],
            ann["y1"],
            ann["x2"],
            ann["y2"],
        )
        
        smoothed_ann = ann.copy()
        smoothed_ann["x1"] = round(x1, 2)
        smoothed_ann["y1"] = round(y1, 2)
        smoothed_ann["x2"] = round(x2, 2)
        smoothed_ann["y2"] = round(y2, 2)
        result.append(smoothed_ann)
    
    return result
