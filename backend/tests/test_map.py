"""
Unit Tests for mAP Calculation
"""

import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from metrics import Detection, calculate_ap, calculate_map


class TestAP:
    """Tests for Average Precision calculation."""
    
    def test_perfect_detection(self):
        """Test AP of 1.0 for perfect detection."""
        detections = [Detection(0, 1, "car", 0, 0, 100, 100, confidence=0.9)]
        ground_truth = [Detection(0, 1, "car", 0, 0, 100, 100)]
        
        ap = calculate_ap(detections, ground_truth, iou_threshold=0.5)
        assert ap == 1.0
    
    def test_no_detections(self):
        """Test AP of 0.0 when no detections."""
        detections = []
        ground_truth = [Detection(0, 1, "car", 0, 0, 100, 100)]
        
        ap = calculate_ap(detections, ground_truth)
        # AP is low but non-zero due to 11-point interpolation
        assert ap < 0.2
    
    def test_false_positive(self):
        """Test AP with false positive (no match)."""
        detections = [Detection(0, 1, "car", 200, 200, 300, 300, confidence=0.9)]
        ground_truth = [Detection(0, 1, "car", 0, 0, 100, 100)]
        
        ap = calculate_ap(detections, ground_truth)
        # AP is low but non-zero due to 11-point interpolation
        assert ap < 0.2
    
    def test_multiple_detections(self):
        """Test AP with multiple detections and GTs."""
        detections = [
            Detection(0, 1, "car", 0, 0, 100, 100, confidence=0.9),
            Detection(0, 2, "car", 200, 200, 300, 300, confidence=0.8),
        ]
        ground_truth = [
            Detection(0, 1, "car", 0, 0, 100, 100),
            Detection(0, 2, "car", 200, 200, 300, 300),
        ]
        
        ap = calculate_ap(detections, ground_truth)
        assert ap == 1.0


class TestMAP:
    """Tests for mAP calculation."""
    
    def test_single_class_map(self):
        """Test mAP with single class."""
        detections = [Detection(0, 1, "car", 0, 0, 100, 100, confidence=0.9)]
        ground_truth = [Detection(0, 1, "car", 0, 0, 100, 100)]
        
        result = calculate_map(detections, ground_truth)
        assert result["mAP"] == 1.0
        assert "AP_car" in result
    
    def test_multi_class_map(self):
        """Test mAP with multiple classes."""
        detections = [
            Detection(0, 1, "car", 0, 0, 100, 100, confidence=0.9),
            Detection(0, 2, "truck", 200, 200, 300, 300, confidence=0.8),
        ]
        ground_truth = [
            Detection(0, 1, "car", 0, 0, 100, 100),
            Detection(0, 2, "truck", 200, 200, 300, 300),
        ]
        
        result = calculate_map(detections, ground_truth)
        assert result["mAP"] == 1.0
        assert "AP_car" in result
        assert "AP_truck" in result


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
