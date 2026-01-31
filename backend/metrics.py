"""
Validation Metrics Module
- IoU calculation
- mAP calculation
- MOTA/MOTP tracking metrics
- ID switch detection
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass


@dataclass
class Detection:
    """Single detection/annotation."""
    frame_id: int
    track_id: int
    class_name: str
    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float = 1.0


@dataclass
class TrackingMetrics:
    """MOTA/MOTP and related metrics."""
    mota: float  # Multi-Object Tracking Accuracy
    motp: float  # Multi-Object Tracking Precision
    num_switches: int  # ID switches
    num_fragmentations: int  # Track fragmentations
    num_false_positives: int
    num_false_negatives: int
    num_matches: int
    mostly_tracked: int  # Tracks with >80% coverage
    mostly_lost: int  # Tracks with <20% coverage


# -----------------------------
# IoU Calculation
# -----------------------------
def calculate_iou(box1: Tuple[float, float, float, float], 
                  box2: Tuple[float, float, float, float]) -> float:
    """
    Calculate Intersection over Union between two boxes.
    
    Args:
        box1, box2: (x1, y1, x2, y2) format
    
    Returns:
        IoU value between 0 and 1
    """
    x1_1, y1_1, x2_1, y2_1 = box1
    x1_2, y1_2, x2_2, y2_2 = box2
    
    # Intersection
    x1_i = max(x1_1, x1_2)
    y1_i = max(y1_1, y1_2)
    x2_i = min(x2_1, x2_2)
    y2_i = min(y2_1, y2_2)
    
    if x2_i < x1_i or y2_i < y1_i:
        return 0.0
    
    intersection = (x2_i - x1_i) * (y2_i - y1_i)
    
    # Union
    area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
    area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
    union = area1 + area2 - intersection
    
    return intersection / union if union > 0 else 0.0


def calculate_iou_matrix(
    detections: List[Detection],
    ground_truth: List[Detection],
) -> np.ndarray:
    """Calculate IoU matrix between detections and ground truth."""
    n_det = len(detections)
    n_gt = len(ground_truth)
    
    iou_matrix = np.zeros((n_det, n_gt))
    
    for i, det in enumerate(detections):
        for j, gt in enumerate(ground_truth):
            iou_matrix[i, j] = calculate_iou(
                (det.x1, det.y1, det.x2, det.y2),
                (gt.x1, gt.y1, gt.x2, gt.y2)
            )
    
    return iou_matrix


# -----------------------------
# mAP Calculation
# -----------------------------
def calculate_ap(
    detections: List[Detection],
    ground_truth: List[Detection],
    iou_threshold: float = 0.5,
) -> float:
    """
    Calculate Average Precision for a single class.
    
    Args:
        detections: Sorted by confidence (descending)
        ground_truth: Ground truth annotations
        iou_threshold: IoU threshold for matching
    
    Returns:
        AP value
    """
    if not ground_truth:
        return 0.0 if detections else 1.0
    
    if not detections:
        return 0.0
    
    # Sort by confidence
    detections = sorted(detections, key=lambda x: x.confidence, reverse=True)
    
    n_gt = len(ground_truth)
    gt_matched = [False] * n_gt
    
    tp = np.zeros(len(detections))
    fp = np.zeros(len(detections))
    
    for i, det in enumerate(detections):
        best_iou = 0.0
        best_gt_idx = -1
        
        for j, gt in enumerate(ground_truth):
            if gt_matched[j]:
                continue
            
            iou = calculate_iou(
                (det.x1, det.y1, det.x2, det.y2),
                (gt.x1, gt.y1, gt.x2, gt.y2)
            )
            
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = j
        
        if best_iou >= iou_threshold and best_gt_idx >= 0:
            tp[i] = 1
            gt_matched[best_gt_idx] = True
        else:
            fp[i] = 1
    
    # Calculate precision/recall curve
    tp_cumsum = np.cumsum(tp)
    fp_cumsum = np.cumsum(fp)
    
    recalls = tp_cumsum / n_gt
    precisions = tp_cumsum / (tp_cumsum + fp_cumsum)
    
    # Interpolate precision
    precisions = np.concatenate([[1], precisions])
    recalls = np.concatenate([[0], recalls])
    
    for i in range(len(precisions) - 2, -1, -1):
        precisions[i] = max(precisions[i], precisions[i + 1])
    
    # Calculate AP using 11-point interpolation
    ap = 0.0
    for t in np.linspace(0, 1, 11):
        p = precisions[recalls >= t]
        ap += p.max() if len(p) > 0 else 0
    
    return ap / 11


def calculate_map(
    detections: List[Detection],
    ground_truth: List[Detection],
    iou_threshold: float = 0.5,
) -> Dict[str, float]:
    """
    Calculate mAP across all classes.
    
    Returns:
        Dict with per-class AP and overall mAP
    """
    # Group by class
    classes = set(d.class_name for d in ground_truth)
    
    results = {}
    aps = []
    
    for cls in classes:
        cls_det = [d for d in detections if d.class_name == cls]
        cls_gt = [g for g in ground_truth if g.class_name == cls]
        
        ap = calculate_ap(cls_det, cls_gt, iou_threshold)
        results[f"AP_{cls}"] = round(ap, 4)
        aps.append(ap)
    
    results["mAP"] = round(np.mean(aps) if aps else 0.0, 4)
    return results


# -----------------------------
# MOTA Calculation
# -----------------------------
def calculate_mota(
    predictions: Dict[int, List[Detection]],  # frame_id -> detections
    ground_truth: Dict[int, List[Detection]],  # frame_id -> gt
    iou_threshold: float = 0.5,
) -> TrackingMetrics:
    """
    Calculate MOTA (Multi-Object Tracking Accuracy).
    
    MOTA = 1 - (FN + FP + ID_switches) / total_gt
    
    Args:
        predictions: Dict mapping frame_id to list of predictions
        ground_truth: Dict mapping frame_id to list of ground truth
    
    Returns:
        TrackingMetrics object
    """
    total_gt = 0
    total_fp = 0
    total_fn = 0
    total_matches = 0
    total_switches = 0
    total_iou = 0.0
    
    # Track ID mapping: gt_id -> predicted_id
    id_mapping: Dict[int, int] = {}
    
    all_frames = sorted(set(predictions.keys()) | set(ground_truth.keys()))
    
    for frame_id in all_frames:
        preds = predictions.get(frame_id, [])
        gts = ground_truth.get(frame_id, [])
        
        total_gt += len(gts)
        
        if not gts:
            total_fp += len(preds)
            continue
        
        if not preds:
            total_fn += len(gts)
            continue
        
        # Calculate IoU matrix
        iou_matrix = calculate_iou_matrix(preds, gts)
        
        # Greedy matching
        matched_preds = set()
        matched_gts = set()
        
        while True:
            max_iou = iou_threshold
            best_pred, best_gt = -1, -1
            
            for i in range(len(preds)):
                if i in matched_preds:
                    continue
                for j in range(len(gts)):
                    if j in matched_gts:
                        continue
                    if iou_matrix[i, j] > max_iou:
                        max_iou = iou_matrix[i, j]
                        best_pred, best_gt = i, j
            
            if best_pred < 0:
                break
            
            matched_preds.add(best_pred)
            matched_gts.add(best_gt)
            total_matches += 1
            total_iou += max_iou
            
            # Check for ID switch
            gt_id = gts[best_gt].track_id
            pred_id = preds[best_pred].track_id
            
            if gt_id in id_mapping:
                if id_mapping[gt_id] != pred_id:
                    total_switches += 1
                    id_mapping[gt_id] = pred_id
            else:
                id_mapping[gt_id] = pred_id
        
        # Count FP and FN
        total_fp += len(preds) - len(matched_preds)
        total_fn += len(gts) - len(matched_gts)
    
    # Calculate metrics
    mota = 1 - (total_fn + total_fp + total_switches) / max(total_gt, 1)
    motp = total_iou / max(total_matches, 1)
    
    return TrackingMetrics(
        mota=round(mota, 4),
        motp=round(motp, 4),
        num_switches=total_switches,
        num_fragmentations=0,  # Would need temporal analysis
        num_false_positives=total_fp,
        num_false_negatives=total_fn,
        num_matches=total_matches,
        mostly_tracked=0,
        mostly_lost=0,
    )


# -----------------------------
# ID Switch Calculation
# -----------------------------
def calculate_id_switches(
    predictions: Dict[int, List[Detection]],
    ground_truth: Dict[int, List[Detection]],
    iou_threshold: float = 0.5,
) -> Dict[str, any]:
    """
    Calculate ID switch rate and related statistics.
    
    Returns:
        Dict with switch count and rate per 1000 frames
    """
    metrics = calculate_mota(predictions, ground_truth, iou_threshold)
    
    total_frames = len(set(predictions.keys()) | set(ground_truth.keys()))
    switch_rate = metrics.num_switches / max(total_frames / 1000, 1)
    
    return {
        "id_switches": metrics.num_switches,
        "switches_per_1000_frames": round(switch_rate, 2),
        "total_frames": total_frames,
    }
