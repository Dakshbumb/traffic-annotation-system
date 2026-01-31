from datetime import datetime
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, ConfigDict


class VideoBase(BaseModel):
    original_filename: str


class VideoCreate(VideoBase):
    filename: str


class Video(VideoBase):
    id: int
    filename: str
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)


class AnnotationBase(BaseModel):
    video_id: int
    frame_index: int
    class_label: str
    x1: float
    y1: float
    x2: float
    y2: float
    track_id: Optional[int] = None
    extra_meta: Optional[Dict[str, Any]] = None


class AnnotationCreate(AnnotationBase):
    pass


class Annotation(AnnotationBase):
    id: int
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)


class AnnotationUpdate(BaseModel):
    """Schema for updating an annotation - all fields optional."""
    x1: Optional[float] = None
    y1: Optional[float] = None
    x2: Optional[float] = None
    y2: Optional[float] = None
    class_label: Optional[str] = None
    track_id: Optional[int] = None


class AnnotationList(BaseModel):
    video_id: int
    annotations: List[Annotation]

class BulkAnnotationCreate(BaseModel):
    video_id: int
    annotations: List[AnnotationCreate]

class InterpolationRequest(BaseModel):
    video_id: int
    track_id: int
    start_frame: int
    end_frame: int
    overwrite: bool = False

class AutolabelRequest(BaseModel):
    video_id: int
    model: str = "yolov8n"
    confidence_threshold: float = 0.4
    frame_stride: int = 1
    start_frame: int = 0
    end_frame: int | None = None
    overwrite: bool = True


class AutolabelJobBase(BaseModel):
    job_id: str
    video_id: int
    status: str
    progress: float
    created_at: datetime
    updated_at: datetime
    params: dict | None = None
    error_message: str | None = None


class AutolabelJobList(BaseModel):
    jobs: list[AutolabelJobBase]


# ------------------- Analytics Lines -------------------

class AnalyticsLineBase(BaseModel):
    name: str = "Line 1"
    line_type: str = "count"  # count | calibration
    points: List[List[float]] # [[x1, y1], [x2, y2]]
    meta_data: Optional[Dict[str, Any]] = None

class AnalyticsLineCreate(AnalyticsLineBase):
    video_id: int

class AnalyticsLine(AnalyticsLineBase):
    id: int
    video_id: int
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)

