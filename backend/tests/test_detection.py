"""
Unit Tests for Detection Module
"""

import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from metrics import calculate_iou, Detection


class TestIoU:
    """Tests for IoU calculation."""
    
    def test_perfect_overlap(self):
        """Test IoU of 1.0 for identical boxes."""
        box = (0, 0, 100, 100)
        assert calculate_iou(box, box) == 1.0
    
    def test_no_overlap(self):
        """Test IoU of 0.0 for non-overlapping boxes."""
        box1 = (0, 0, 50, 50)
        box2 = (100, 100, 150, 150)
        assert calculate_iou(box1, box2) == 0.0
    
    def test_partial_overlap(self):
        """Test IoU for partially overlapping boxes."""
        box1 = (0, 0, 100, 100)
        box2 = (50, 50, 150, 150)
        # Intersection: 50x50 = 2500
        # Union: 2*10000 - 2500 = 17500
        # IoU = 2500 / 17500 ≈ 0.143
        iou = calculate_iou(box1, box2)
        assert 0.14 < iou < 0.15
    
    def test_contained_box(self):
        """Test IoU when one box contains another."""
        outer = (0, 0, 100, 100)
        inner = (25, 25, 75, 75)
        # Intersection: 50x50 = 2500
        # Union: 10000
        # IoU = 0.25
        iou = calculate_iou(outer, inner)
        assert iou == 0.25


class TestDetection:
    """Tests for Detection dataclass."""
    
    def test_detection_creation(self):
        """Test basic detection creation."""
        det = Detection(
            frame_id=0,
            track_id=1,
            class_name="car",
            x1=10, y1=20, x2=100, y2=200,
            confidence=0.95
        )
        assert det.frame_id == 0
        assert det.track_id == 1
        assert det.class_name == "car"
        assert det.confidence == 0.95
    
    def test_default_confidence(self):
        """Test default confidence value."""
        det = Detection(
            frame_id=0, track_id=1, class_name="car",
            x1=0, y1=0, x2=100, y2=100
        )
        assert det.confidence == 1.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
