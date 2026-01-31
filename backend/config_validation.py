"""
System Configuration Validation
Pydantic-based config validation with YAML support.
"""

from typing import Dict, List, Optional
from pydantic import BaseModel, Field, field_validator
import os

# Import current config values as defaults
from config import (
    AUTO_DEFAULT_CONF_TH,
    AUTO_DEFAULT_NMS_IOU,
    DEEPSORT_MAX_AGE,
    DEEPSORT_N_INIT,
    DEEPSORT_MAX_COSINE_DIST,
    DEEPSORT_NN_BUDGET,
    MIN_BOX_AREA,
    ASPECT_RATIO_MIN,
    ASPECT_RATIO_MAX,
    EDGE_MARGIN,
    ENABLE_CLAHE,
    CLAHE_CLIP_LIMIT,
)


class DetectionConfig(BaseModel):
    """Detection model configuration."""
    model: str = Field(default="yolov8s.pt", description="YOLO model name or path")
    confidence_threshold: float = Field(default=AUTO_DEFAULT_CONF_TH, ge=0.0, le=1.0)
    nms_threshold: float = Field(default=AUTO_DEFAULT_NMS_IOU, ge=0.0, le=1.0)
    input_size: List[int] = Field(default=[640, 640], min_length=2, max_length=2)
    classes: List[int] = Field(default=[2, 3, 5, 7], description="COCO class IDs to detect")
    
    @field_validator('input_size')
    @classmethod
    def validate_input_size(cls, v):
        if any(dim < 32 or dim > 2048 for dim in v):
            raise ValueError("Input size must be between 32 and 2048")
        return v


class TrackingConfig(BaseModel):
    """DeepSORT tracking configuration."""
    max_age: int = Field(default=DEEPSORT_MAX_AGE, ge=1, le=200)
    min_hits: int = Field(default=DEEPSORT_N_INIT, ge=1, le=20)
    iou_threshold: float = Field(default=0.3, ge=0.0, le=1.0)
    max_cosine_distance: float = Field(default=DEEPSORT_MAX_COSINE_DIST, ge=0.0, le=1.0)
    nn_budget: int = Field(default=DEEPSORT_NN_BUDGET, ge=10, le=500)


class PreprocessingConfig(BaseModel):
    """Frame preprocessing configuration."""
    enable_clahe: bool = Field(default=ENABLE_CLAHE)
    clahe_clip_limit: float = Field(default=CLAHE_CLIP_LIMIT, ge=1.0, le=10.0)
    clahe_tile_size: int = Field(default=8, ge=2, le=32)


class PostprocessingConfig(BaseModel):
    """Detection postprocessing configuration."""
    min_box_area: int = Field(default=MIN_BOX_AREA, ge=0)
    min_aspect_ratio: float = Field(default=ASPECT_RATIO_MIN, ge=0.0)
    max_aspect_ratio: float = Field(default=ASPECT_RATIO_MAX, ge=1.0)
    edge_margin: int = Field(default=EDGE_MARGIN, ge=0)
    
    @field_validator('max_aspect_ratio')
    @classmethod
    def validate_aspect_ratio(cls, v, info):
        min_ratio = info.data.get('min_aspect_ratio', 0)
        if v <= min_ratio:
            raise ValueError("max_aspect_ratio must be greater than min_aspect_ratio")
        return v


class AnnotationConfig(BaseModel):
    """Annotation export configuration."""
    format: str = Field(default="COCO", pattern="^(COCO|YOLO|VOC|JSON|CSV)$")
    include_confidence: bool = Field(default=True)
    include_tracking_id: bool = Field(default=True)
    temporal_smoothing: bool = Field(default=True)
    smoothing_alpha: float = Field(default=0.7, ge=0.0, le=1.0)


class MonitoringConfig(BaseModel):
    """System monitoring configuration."""
    fps_log_interval: int = Field(default=100, ge=10, le=1000)
    fps_alert_threshold: float = Field(default=15.0, ge=1.0)
    fps_alert_duration: float = Field(default=10.0, ge=1.0)
    memory_warn_threshold: float = Field(default=0.85, ge=0.5, le=1.0)


class SystemConfig(BaseModel):
    """Complete system configuration."""
    detection: DetectionConfig = DetectionConfig()
    tracking: TrackingConfig = TrackingConfig()
    preprocessing: PreprocessingConfig = PreprocessingConfig()
    postprocessing: PostprocessingConfig = PostprocessingConfig()
    annotation: AnnotationConfig = AnnotationConfig()
    monitoring: MonitoringConfig = MonitoringConfig()
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return self.model_dump()
    
    @classmethod
    def from_yaml(cls, path: str) -> "SystemConfig":
        """Load config from YAML file."""
        try:
            import yaml
            with open(path, "r") as f:
                data = yaml.safe_load(f)
            return cls(**data) if data else cls()
        except ImportError:
            raise ImportError("PyYAML required: pip install pyyaml")
        except FileNotFoundError:
            return cls()
    
    def to_yaml(self, path: str) -> str:
        """Save config to YAML file."""
        try:
            import yaml
            with open(path, "w") as f:
                yaml.dump(self.to_dict(), f, default_flow_style=False)
            return path
        except ImportError:
            raise ImportError("PyYAML required: pip install pyyaml")


def validate_config(config_dict: dict) -> Dict[str, any]:
    """
    Validate a configuration dictionary.
    
    Returns:
        Dict with 'valid' boolean and 'errors' list
    """
    try:
        SystemConfig(**config_dict)
        return {"valid": True, "errors": []}
    except Exception as e:
        return {"valid": False, "errors": [str(e)]}


# Default config instance
default_config = SystemConfig()
