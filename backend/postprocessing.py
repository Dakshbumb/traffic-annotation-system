"""
Post-processing filters for detection pipeline.
Filters detections by area, aspect ratio, and edge proximity.
"""

from config import (
    MIN_BOX_AREA,
    ASPECT_RATIO_MIN,
    ASPECT_RATIO_MAX,
    EDGE_MARGIN,
    CLASS_CONF_THRESHOLDS,
    AUTO_DEFAULT_CONF_TH,
)


def filter_by_area(x1: float, y1: float, x2: float, y2: float) -> bool:
    """
    Check if detection meets minimum area threshold.
    
    Args:
        x1, y1, x2, y2: Bounding box coordinates
    
    Returns:
        True if detection should be KEPT, False to discard
    """
    area = (x2 - x1) * (y2 - y1)
    return area >= MIN_BOX_AREA


def filter_by_aspect_ratio(x1: float, y1: float, x2: float, y2: float) -> bool:
    """
    Check if detection has valid aspect ratio.
    
    Args:
        x1, y1, x2, y2: Bounding box coordinates
    
    Returns:
        True if detection should be KEPT, False to discard
    """
    w = x2 - x1
    h = y2 - y1
    
    if w <= 0:
        return False
    
    ratio = h / w
    return ASPECT_RATIO_MIN <= ratio <= ASPECT_RATIO_MAX


def filter_edge_detections(
    x1: float, y1: float, x2: float, y2: float,
    frame_width: int, frame_height: int
) -> bool:
    """
    Check if detection is too close to frame edges.
    
    Args:
        x1, y1, x2, y2: Bounding box coordinates
        frame_width, frame_height: Frame dimensions
    
    Returns:
        True if detection should be KEPT, False to discard
    """
    # Check if any edge of bbox is within EDGE_MARGIN of frame boundary
    if x1 < EDGE_MARGIN or y1 < EDGE_MARGIN:
        return False
    if x2 > (frame_width - EDGE_MARGIN) or y2 > (frame_height - EDGE_MARGIN):
        return False
    
    return True


def get_class_confidence_threshold(class_name: str) -> float:
    """
    Get per-class confidence threshold.
    
    Args:
        class_name: Name of the class
    
    Returns:
        Confidence threshold for this class
    """
    return CLASS_CONF_THRESHOLDS.get(class_name, AUTO_DEFAULT_CONF_TH)


def apply_all_filters(
    x1: float, y1: float, x2: float, y2: float,
    frame_width: int, frame_height: int,
    confidence: float = None,
    class_name: str = None
) -> bool:
    """
    Apply all post-processing filters to a detection.
    
    Args:
        x1, y1, x2, y2: Bounding box coordinates
        frame_width, frame_height: Frame dimensions
        confidence: Detection confidence (optional)
        class_name: Class name for per-class threshold (optional)
    
    Returns:
        True if detection passes ALL filters, False otherwise
    """
    # 1. Check confidence (if provided)
    if confidence is not None and class_name is not None:
        threshold = get_class_confidence_threshold(class_name)
        if confidence < threshold:
            return False
    
    # 2. Check minimum area
    if not filter_by_area(x1, y1, x2, y2):
        return False
    
    # 3. Check aspect ratio
    if not filter_by_aspect_ratio(x1, y1, x2, y2):
        return False
    
    # 4. Check edge proximity
    if not filter_edge_detections(x1, y1, x2, y2, frame_width, frame_height):
        return False
    
    return True
