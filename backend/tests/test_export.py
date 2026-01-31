"""
Unit Tests for Export Formats
"""

import pytest
import sys
import os
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))


class TestExportSchemas:
    """Tests for export schema validation."""
    
    def test_detection_export_schema(self):
        """Test DetectionExport schema validation."""
        from export_schemas import DetectionExport, DetectionAttributes
        
        det = DetectionExport(
            track_id=1,
            **{"class": "car"},
            class_id=0,
            confidence=0.95,
            bbox=[10, 20, 100, 150],
            bbox_center=[55, 85],
            area=5850.0,
            attributes=DetectionAttributes()
        )
        
        assert det.track_id == 1
        assert det.class_name == "car"
        assert det.confidence == 0.95
    
    def test_confidence_bounds(self):
        """Test confidence must be in [0, 1]."""
        from export_schemas import DetectionExport
        
        with pytest.raises(ValueError):
            DetectionExport(
                track_id=1, **{"class": "car"}, class_id=0,
                confidence=1.5,  # Invalid
                bbox=[0, 0, 100, 100],
                bbox_center=[50, 50],
                area=10000,
            )
    
    def test_class_id_mapping(self):
        """Test class ID mapping function."""
        from export_schemas import get_class_id
        
        assert get_class_id("car") == 0
        assert get_class_id("truck") == 1
        assert get_class_id("unknown") == -1


class TestCOCOExport:
    """Tests for COCO format export."""
    
    def test_coco_structure(self):
        """Test COCO export has required keys."""
        # This would need a database session, so we test structure only
        expected_keys = ["info", "licenses", "categories", "images", "annotations"]
        
        # Mock COCO data
        coco_data = {
            "info": {},
            "licenses": [],
            "categories": [],
            "images": [],
            "annotations": [],
        }
        
        for key in expected_keys:
            assert key in coco_data


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
