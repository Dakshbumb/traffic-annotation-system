"""
Performance Monitoring Module
- FPS tracking and logging
- Memory usage monitoring
- ID switch detection
- Alert system for performance issues
"""

import time
import logging
import threading
from collections import deque
from typing import Optional, Callable
from dataclasses import dataclass, field

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s'
)
logger = logging.getLogger("monitoring")


@dataclass
class PerformanceStats:
    """Container for performance statistics."""
    total_frames: int = 0
    total_detections: int = 0
    id_switches: int = 0
    failed_detections: int = 0
    failed_tracks: int = 0
    tracker_resets: int = 0


class PerformanceMonitor:
    """
    Real-time performance monitoring for detection/tracking pipeline.
    
    Features:
    - FPS calculation and logging (every N frames)
    - Low FPS alert detection
    - Memory usage monitoring
    - ID switch tracking
    """
    
    def __init__(
        self,
        log_interval: int = 100,
        fps_alert_threshold: float = 15.0,
        fps_alert_duration: float = 10.0,
        memory_warn_threshold: float = 0.85,
    ):
        self.log_interval = log_interval
        self.fps_alert_threshold = fps_alert_threshold
        self.fps_alert_duration = fps_alert_duration
        self.memory_warn_threshold = memory_warn_threshold
        
        # Timing
        self.start_time: Optional[float] = None
        self.frame_times: deque = deque(maxlen=100)  # Rolling window
        self.last_log_frame: int = 0
        
        # Stats
        self.stats = PerformanceStats()
        
        # Low FPS tracking
        self.low_fps_start: Optional[float] = None
        self.fps_alert_triggered: bool = False
        
        # Track ID history for switch detection
        self.prev_track_ids: set = set()
        
        # Callbacks
        self.on_fps_alert: Optional[Callable] = None
        self.on_memory_alert: Optional[Callable] = None
    
    def start(self):
        """Start monitoring session."""
        self.start_time = time.time()
        self.frame_times.clear()
        self.stats = PerformanceStats()
        self.prev_track_ids.clear()
        logger.info("Performance monitoring started")
    
    def frame_start(self):
        """Call at the start of processing a frame."""
        self._frame_start_time = time.time()
    
    def frame_end(self, frame_idx: int, track_ids: set = None):
        """
        Call at the end of processing a frame.
        
        Args:
            frame_idx: Current frame index
            track_ids: Set of active track IDs in this frame
        """
        frame_time = time.time() - self._frame_start_time
        self.frame_times.append(frame_time)
        self.stats.total_frames += 1
        
        # Detect ID switches
        if track_ids is not None:
            self._check_id_switches(track_ids)
        
        # Log FPS periodically
        if self.stats.total_frames - self.last_log_frame >= self.log_interval:
            self._log_fps()
            self.last_log_frame = self.stats.total_frames
        
        # Check for low FPS
        self._check_fps_alert()
    
    def _check_id_switches(self, current_ids: set):
        """Detect potential ID switches by comparing track IDs."""
        if self.prev_track_ids:
            # Lost tracks that might be ID switches
            lost = self.prev_track_ids - current_ids
            new = current_ids - self.prev_track_ids
            
            # Heuristic: if we lose tracks and gain new ones simultaneously,
            # it might indicate ID switches
            if len(lost) > 0 and len(new) > 0:
                potential_switches = min(len(lost), len(new))
                self.stats.id_switches += potential_switches
        
        self.prev_track_ids = current_ids.copy()
    
    def _log_fps(self):
        """Log current FPS statistics."""
        if not self.frame_times:
            return
        
        avg_time = sum(self.frame_times) / len(self.frame_times)
        fps = 1.0 / avg_time if avg_time > 0 else 0
        
        elapsed = time.time() - self.start_time if self.start_time else 0
        overall_fps = self.stats.total_frames / elapsed if elapsed > 0 else 0
        
        logger.info(
            f"[Frame {self.stats.total_frames}] "
            f"FPS: {fps:.1f} (avg: {overall_fps:.1f}) | "
            f"Detections: {self.stats.total_detections} | "
            f"ID Switches: {self.stats.id_switches}"
        )
    
    def _check_fps_alert(self):
        """Check for sustained low FPS and trigger alert."""
        if not self.frame_times:
            return
        
        avg_time = sum(self.frame_times) / len(self.frame_times)
        fps = 1.0 / avg_time if avg_time > 0 else 0
        
        if fps < self.fps_alert_threshold:
            if self.low_fps_start is None:
                self.low_fps_start = time.time()
            elif time.time() - self.low_fps_start > self.fps_alert_duration:
                if not self.fps_alert_triggered:
                    logger.warning(
                        f"ALERT: FPS below {self.fps_alert_threshold} "
                        f"for {self.fps_alert_duration}s (current: {fps:.1f})"
                    )
                    self.fps_alert_triggered = True
                    if self.on_fps_alert:
                        self.on_fps_alert(fps)
        else:
            self.low_fps_start = None
            self.fps_alert_triggered = False
    
    def record_detection_failure(self):
        """Record a detection failure."""
        self.stats.failed_detections += 1
        logger.warning(f"Detection failure #{self.stats.failed_detections}")
    
    def record_tracking_failure(self):
        """Record a tracking failure."""
        self.stats.failed_tracks += 1
        logger.warning(f"Tracking failure #{self.stats.failed_tracks}")
    
    def record_tracker_reset(self):
        """Record a tracker reset event."""
        self.stats.tracker_resets += 1
        self.prev_track_ids.clear()
        logger.warning(f"Tracker reset #{self.stats.tracker_resets}")
    
    def record_detections(self, count: int):
        """Record number of detections in a frame."""
        self.stats.total_detections += count
    
    def get_summary(self) -> dict:
        """Get summary of monitoring statistics."""
        elapsed = time.time() - self.start_time if self.start_time else 0
        avg_fps = self.stats.total_frames / elapsed if elapsed > 0 else 0
        
        return {
            "total_frames": self.stats.total_frames,
            "elapsed_seconds": round(elapsed, 2),
            "average_fps": round(avg_fps, 2),
            "total_detections": self.stats.total_detections,
            "id_switches": self.stats.id_switches,
            "id_switches_per_1000": round(self.stats.id_switches / max(1, self.stats.total_frames / 1000), 2),
            "failed_detections": self.stats.failed_detections,
            "failed_tracks": self.stats.failed_tracks,
            "tracker_resets": self.stats.tracker_resets,
        }
    
    def stop(self):
        """Stop monitoring and log final summary."""
        summary = self.get_summary()
        logger.info(f"Monitoring stopped. Summary: {summary}")
        return summary


def get_memory_usage() -> dict:
    """Get current memory usage statistics."""
    try:
        import psutil
        process = psutil.Process()
        mem_info = process.memory_info()
        
        return {
            "rss_mb": round(mem_info.rss / 1024 / 1024, 2),
            "vms_mb": round(mem_info.vms / 1024 / 1024, 2),
            "percent": round(process.memory_percent(), 2),
        }
    except ImportError:
        return {"error": "psutil not installed"}
    except Exception as e:
        return {"error": str(e)}


def get_gpu_memory_usage() -> dict:
    """Get GPU memory usage (if available)."""
    try:
        import torch
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1024 / 1024
            reserved = torch.cuda.memory_reserved() / 1024 / 1024
            total = torch.cuda.get_device_properties(0).total_memory / 1024 / 1024
            
            return {
                "allocated_mb": round(allocated, 2),
                "reserved_mb": round(reserved, 2),
                "total_mb": round(total, 2),
                "utilization": round(allocated / total, 4) if total > 0 else 0,
            }
        return {"available": False}
    except ImportError:
        return {"error": "torch not installed"}
    except Exception as e:
        return {"error": str(e)}
