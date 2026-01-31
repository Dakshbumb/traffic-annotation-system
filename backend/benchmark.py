"""
Benchmark Runner
- Runs detection/tracking on test videos
- Measures FPS, memory usage
- Compares against ground truth for mAP/MOTA
"""

import os
import sys
import time
import json
import argparse
from datetime import datetime
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import cv2
import numpy as np

from metrics import Detection, calculate_map, calculate_mota, calculate_iou
from monitoring import PerformanceMonitor, get_memory_usage


@dataclass
class BenchmarkResult:
    """Container for benchmark results."""
    video_name: str
    total_frames: int
    processed_frames: int
    total_detections: int
    
    # Performance
    average_fps: float
    min_fps: float
    max_fps: float
    total_time_seconds: float
    
    # Memory
    peak_memory_mb: float
    avg_memory_mb: float
    
    # Accuracy (if ground truth available)
    mAP: Optional[float] = None
    mota: Optional[float] = None
    motp: Optional[float] = None
    id_switches: Optional[int] = None
    false_positives: Optional[int] = None
    false_negatives: Optional[int] = None


class BenchmarkRunner:
    """
    Runs benchmarks on the detection/tracking pipeline.
    """
    
    def __init__(self, output_dir: str = "benchmark_results"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        self.results: List[BenchmarkResult] = []
    
    def run_detection_benchmark(
        self,
        video_path: str,
        ground_truth_path: Optional[str] = None,
        max_frames: int = 1000,
        confidence_threshold: float = 0.25,
    ) -> BenchmarkResult:
        """
        Run detection benchmark on a video.
        
        Args:
            video_path: Path to video file
            ground_truth_path: Optional path to ground truth JSON
            max_frames: Maximum frames to process
            confidence_threshold: Detection confidence threshold
        
        Returns:
            BenchmarkResult object
        """
        from ultralytics import YOLO
        from config import YOLO_MODEL_NAME, YOLO_COCO_CLASS_MAP, ALLOWED_CLASSES
        from preprocessing import preprocess_frame
        from postprocessing import apply_all_filters, get_class_confidence_threshold
        
        print(f"Running benchmark on: {video_path}")
        
        # Load model
        yolo = YOLO(YOLO_MODEL_NAME)
        yolo.to("cpu")
        
        # Open video
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps_video = cap.get(cv2.CAP_PROP_FPS) or 30
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Load ground truth if available
        ground_truth: Dict[int, List[Detection]] = {}
        if ground_truth_path and os.path.exists(ground_truth_path):
            ground_truth = self._load_ground_truth(ground_truth_path)
        
        # Metrics tracking
        monitor = PerformanceMonitor(log_interval=100)
        monitor.start()
        
        frame_times = []
        memory_samples = []
        all_detections: Dict[int, List[Detection]] = {}
        
        frame_idx = 0
        processed = 0
        
        while processed < max_frames:
            ret, frame = cap.read()
            if not ret:
                break
            
            monitor.frame_start()
            start_time = time.time()
            
            # Preprocess
            processed_frame, scale = preprocess_frame(frame)
            
            # Detect
            results = yolo(processed_frame, verbose=False, conf=confidence_threshold)[0]
            
            # Parse detections
            frame_detections = []
            for box in results.boxes:
                cls_idx = int(box.cls[0])
                class_name = YOLO_COCO_CLASS_MAP.get(cls_idx)
                
                if class_name is None or class_name not in ALLOWED_CLASSES:
                    continue
                
                conf = float(box.conf[0])
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                
                if scale < 1.0:
                    x1, y1, x2, y2 = x1/scale, y1/scale, x2/scale, y2/scale
                
                if apply_all_filters(x1, y1, x2, y2, frame_width, frame_height, conf, class_name):
                    frame_detections.append(Detection(
                        frame_id=frame_idx,
                        track_id=-1,
                        class_name=class_name,
                        x1=x1, y1=y1, x2=x2, y2=y2,
                        confidence=conf
                    ))
            
            all_detections[frame_idx] = frame_detections
            
            # Record metrics
            frame_time = time.time() - start_time
            frame_times.append(frame_time)
            
            mem = get_memory_usage()
            if "rss_mb" in mem:
                memory_samples.append(mem["rss_mb"])
            
            monitor.frame_end(frame_idx)
            
            frame_idx += 1
            processed += 1
        
        cap.release()
        summary = monitor.stop()
        
        # Calculate accuracy if ground truth available
        mAP_result = None
        mota_result = None
        
        if ground_truth:
            # Flatten detections for mAP
            flat_dets = [d for dets in all_detections.values() for d in dets]
            flat_gt = [g for gts in ground_truth.values() for g in gts]
            
            mAP_result = calculate_map(flat_dets, flat_gt)
            mota_result = calculate_mota(all_detections, ground_truth)
        
        # Build result
        fps_values = [1/t for t in frame_times if t > 0]
        total_dets = sum(len(d) for d in all_detections.values())
        
        result = BenchmarkResult(
            video_name=os.path.basename(video_path),
            total_frames=total_frames,
            processed_frames=processed,
            total_detections=total_dets,
            average_fps=np.mean(fps_values) if fps_values else 0,
            min_fps=min(fps_values) if fps_values else 0,
            max_fps=max(fps_values) if fps_values else 0,
            total_time_seconds=sum(frame_times),
            peak_memory_mb=max(memory_samples) if memory_samples else 0,
            avg_memory_mb=np.mean(memory_samples) if memory_samples else 0,
            mAP=mAP_result["mAP"] if mAP_result else None,
            mota=mota_result.mota if mota_result else None,
            motp=mota_result.motp if mota_result else None,
            id_switches=mota_result.num_switches if mota_result else None,
            false_positives=mota_result.num_false_positives if mota_result else None,
            false_negatives=mota_result.num_false_negatives if mota_result else None,
        )
        
        self.results.append(result)
        return result
    
    def _load_ground_truth(self, path: str) -> Dict[int, List[Detection]]:
        """Load ground truth from JSON file."""
        with open(path, "r") as f:
            data = json.load(f)
        
        gt: Dict[int, List[Detection]] = {}
        
        for frame in data.get("frames", []):
            frame_id = frame["frame_id"]
            gt[frame_id] = []
            
            for det in frame.get("detections", []):
                gt[frame_id].append(Detection(
                    frame_id=frame_id,
                    track_id=det.get("track_id", -1),
                    class_name=det.get("class", "object"),
                    x1=det["bbox"][0],
                    y1=det["bbox"][1],
                    x2=det["bbox"][2],
                    y2=det["bbox"][3],
                    confidence=1.0,
                ))
        
        return gt
    
    def save_results(self, filename: str = None) -> str:
        """Save all results to JSON."""
        if filename is None:
            ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            filename = f"benchmark_{ts}.json"
        
        path = os.path.join(self.output_dir, filename)
        
        with open(path, "w") as f:
            json.dump([asdict(r) for r in self.results], f, indent=2)
        
        print(f"Results saved to: {path}")
        return path
    
    def print_summary(self):
        """Print summary of all benchmark results."""
        print("\n" + "=" * 60)
        print("BENCHMARK SUMMARY")
        print("=" * 60)
        
        for r in self.results:
            print(f"\n{r.video_name}")
            print(f"  Frames: {r.processed_frames}/{r.total_frames}")
            print(f"  Detections: {r.total_detections}")
            print(f"  FPS: {r.average_fps:.1f} (min: {r.min_fps:.1f}, max: {r.max_fps:.1f})")
            print(f"  Memory: {r.avg_memory_mb:.1f} MB (peak: {r.peak_memory_mb:.1f} MB)")
            
            if r.mAP is not None:
                print(f"  mAP@0.5: {r.mAP:.4f}")
            if r.mota is not None:
                print(f"  MOTA: {r.mota:.4f}")
                print(f"  ID Switches: {r.id_switches}")


def main():
    parser = argparse.ArgumentParser(description="Run detection/tracking benchmark")
    parser.add_argument("video", help="Path to video file")
    parser.add_argument("--ground-truth", "-gt", help="Path to ground truth JSON")
    parser.add_argument("--max-frames", "-n", type=int, default=1000)
    parser.add_argument("--confidence", "-c", type=float, default=0.25)
    parser.add_argument("--output", "-o", default="benchmark_results")
    
    args = parser.parse_args()
    
    runner = BenchmarkRunner(output_dir=args.output)
    result = runner.run_detection_benchmark(
        args.video,
        ground_truth_path=args.ground_truth,
        max_frames=args.max_frames,
        confidence_threshold=args.confidence,
    )
    
    runner.print_summary()
    runner.save_results()


if __name__ == "__main__":
    main()
