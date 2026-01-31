"""
Unit Tests for Tracking Metrics
"""

import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from metrics import Detection, calculate_mota, calculate_id_switches


class TestMOTA:
    """Tests for MOTA calculation."""
    
    def test_perfect_tracking(self):
        """Test MOTA of 1.0 for perfect tracking."""
        predictions = {
            0: [Detection(0, 1, "car", 0, 0, 100, 100)],
            1: [Detection(1, 1, "car", 10, 10, 110, 110)],
        }
        ground_truth = {
            0: [Detection(0, 1, "car", 0, 0, 100, 100)],
            1: [Detection(1, 1, "car", 10, 10, 110, 110)],
        }
        
        metrics = calculate_mota(predictions, ground_truth, iou_threshold=0.5)
        assert metrics.mota == 1.0
        assert metrics.num_switches == 0
    
    def test_all_false_negatives(self):
        """Test MOTA with no predictions."""
        predictions = {0: [], 1: []}
        ground_truth = {
            0: [Detection(0, 1, "car", 0, 0, 100, 100)],
            1: [Detection(1, 1, "car", 10, 10, 110, 110)],
        }
        
        metrics = calculate_mota(predictions, ground_truth)
        # MOTA is 0 when all FN (1 - FN/GT = 1 - 2/2 = 0)
        assert metrics.mota <= 0
    
    def test_id_switch_detection(self):
        """Test ID switch is detected."""
        predictions = {
            0: [Detection(0, 1, "car", 0, 0, 100, 100)],
            1: [Detection(1, 2, "car", 10, 10, 110, 110)],  # ID switched
        }
        ground_truth = {
            0: [Detection(0, 1, "car", 0, 0, 100, 100)],
            1: [Detection(1, 1, "car", 10, 10, 110, 110)],  # Same GT ID
        }
        
        metrics = calculate_mota(predictions, ground_truth)
        assert metrics.num_switches == 1


class TestIDSwitches:
    """Tests for ID switch rate calculation."""
    
    def test_no_switches(self):
        """Test zero switches with consistent IDs."""
        predictions = {i: [Detection(i, 1, "car", 0, 0, 100, 100)] for i in range(100)}
        ground_truth = {i: [Detection(i, 1, "car", 0, 0, 100, 100)] for i in range(100)}
        
        result = calculate_id_switches(predictions, ground_truth)
        assert result["id_switches"] == 0
    
    def test_switches_per_1000(self):
        """Test switch rate calculation."""
        result = calculate_id_switches({}, {})
        assert "switches_per_1000_frames" in result


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
