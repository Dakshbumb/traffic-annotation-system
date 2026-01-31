"""
Export Schemas - Pydantic models for annotation export with validation.
Matches the user-specified JSON schema requirements.
"""

from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field, field_validator, model_validator
from datetime import datetime


# ------------------------------
# Detection Attributes
# ------------------------------
class DetectionAttributes(BaseModel):
    """Attributes for a single detection."""
    occluded: bool = False
    truncated: bool = False
    difficult: bool = False


# ------------------------------
# Single Detection
# ------------------------------
class DetectionExport(BaseModel):
    """Export schema for a single detection."""
    track_id: int
    class_name: str = Field(alias="class")
    class_id: int
    confidence: float = Field(ge=0.0, le=1.0)
    bbox: List[float] = Field(min_length=4, max_length=4)  # [x1, y1, x2, y2]
    bbox_center: List[float] = Field(min_length=2, max_length=2)  # [cx, cy]
    area: float = Field(ge=0.0)
    attributes: DetectionAttributes = DetectionAttributes()

    model_config = {"populate_by_name": True}

    @field_validator("bbox")
    @classmethod
    def validate_bbox(cls, v):
        if len(v) != 4:
            raise ValueError("bbox must have exactly 4 values [x1, y1, x2, y2]")
        x1, y1, x2, y2 = v
        if x2 < x1 or y2 < y1:
            raise ValueError("Invalid bbox: x2 must be >= x1 and y2 must be >= y1")
        return v


# ------------------------------
# Frame Metadata
# ------------------------------
class FrameMetadata(BaseModel):
    """Metadata for a single frame."""
    frame_width: int = Field(gt=0)
    frame_height: int = Field(gt=0)
    fps: float = Field(gt=0)
    processing_time_ms: Optional[float] = None


# ------------------------------
# Single Frame Export
# ------------------------------
class FrameExport(BaseModel):
    """Export schema for a single frame with all detections."""
    frame_id: int = Field(ge=0)
    timestamp: float = Field(ge=0.0)
    camera_id: str = "default"
    detections: List[DetectionExport] = []
    metadata: FrameMetadata

    @model_validator(mode="after")
    def validate_no_duplicate_track_ids(self):
        """Ensure no duplicate track_ids in the same frame."""
        track_ids = [d.track_id for d in self.detections]
        if len(track_ids) != len(set(track_ids)):
            raise ValueError(f"Duplicate track_ids found in frame {self.frame_id}")
        return self

    @model_validator(mode="after")
    def validate_bbox_in_bounds(self):
        """Ensure all bboxes are within frame dimensions."""
        w, h = self.metadata.frame_width, self.metadata.frame_height
        for det in self.detections:
            x1, y1, x2, y2 = det.bbox
            if x1 < 0 or y1 < 0 or x2 > w or y2 > h:
                # Clamp instead of raising error (common in real data)
                det.bbox = [
                    max(0, min(x1, w)),
                    max(0, min(y1, h)),
                    max(0, min(x2, w)),
                    max(0, min(y2, h)),
                ]
        return self


# ------------------------------
# Full Video Export
# ------------------------------
class VideoExport(BaseModel):
    """Complete export for a video."""
    video_id: int
    video_filename: str
    export_format: str
    export_timestamp: datetime = Field(default_factory=datetime.utcnow)
    total_frames: int = Field(ge=0)
    total_detections: int = Field(ge=0)
    frames: List[FrameExport] = []
    classes: Dict[str, int] = {}  # class_name -> count


# ------------------------------
# Class ID Mapping
# ------------------------------
CLASS_ID_MAP = {
    "car": 0,
    "truck": 1,
    "bus": 2,
    "motorcycle": 3,
    "bicycle": 4,
    "person": 5,
}

def get_class_id(class_name: str) -> int:
    """Get numeric class ID for a class name."""
    return CLASS_ID_MAP.get(class_name.lower(), -1)
