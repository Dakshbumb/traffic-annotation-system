"""
Preprocessing module for traffic detection pipeline.
Handles frame normalization, low-light enhancement, and resizing.
"""

import cv2
import numpy as np

from config import (
    ENABLE_CLAHE,
    CLAHE_CLIP_LIMIT,
    CLAHE_TILE_SIZE,
    YOLO_INPUT_SIZE,
)


def apply_clahe(frame: np.ndarray) -> np.ndarray:
    """
    Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
    for low-light enhancement.
    
    Args:
        frame: BGR image (OpenCV format)
    
    Returns:
        Enhanced BGR image
    """
    if not ENABLE_CLAHE:
        return frame
    
    # Convert to LAB color space
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)
    
    # Apply CLAHE to L channel
    clahe = cv2.createCLAHE(
        clipLimit=CLAHE_CLIP_LIMIT,
        tileGridSize=(CLAHE_TILE_SIZE, CLAHE_TILE_SIZE)
    )
    l_enhanced = clahe.apply(l_channel)
    
    # Merge and convert back to BGR
    lab_enhanced = cv2.merge([l_enhanced, a_channel, b_channel])
    result = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)
    
    return result


def normalize_frame(frame: np.ndarray) -> np.ndarray:
    """
    Normalize frame to RGB and ensure proper dtype.
    YOLO expects RGB, OpenCV reads BGR.
    
    Args:
        frame: BGR image (OpenCV format)
    
    Returns:
        RGB normalized image
    """
    # Convert BGR to RGB
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return rgb


def resize_frame(frame: np.ndarray, target_size: int = None) -> tuple[np.ndarray, float]:
    """
    Resize frame to target size while maintaining aspect ratio.
    Returns the resized frame and the scale factor.
    
    Args:
        frame: Input image
        target_size: Target width (default from config)
    
    Returns:
        Tuple of (resized_frame, scale_factor)
    """
    if target_size is None:
        target_size = YOLO_INPUT_SIZE
    
    h, w = frame.shape[:2]
    
    # Calculate scale to fit target size
    scale = target_size / max(h, w)
    
    if scale >= 1.0:
        # Don't upscale
        return frame, 1.0
    
    new_w = int(w * scale)
    new_h = int(h * scale)
    
    resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    
    return resized, scale


def preprocess_frame(frame: np.ndarray, apply_enhancement: bool = True) -> tuple[np.ndarray, float]:
    """
    Full preprocessing pipeline for a frame.
    
    Args:
        frame: BGR image from OpenCV
        apply_enhancement: Whether to apply CLAHE
    
    Returns:
        Tuple of (preprocessed_frame, scale_factor)
    """
    # 1. Apply low-light enhancement if enabled
    if apply_enhancement:
        frame = apply_clahe(frame)
    
    # 2. Resize to target size
    frame, scale = resize_frame(frame)
    
    # Note: We keep BGR for YOLO (ultralytics handles conversion internally)
    # Only convert to RGB if needed for visualization
    
    return frame, scale
