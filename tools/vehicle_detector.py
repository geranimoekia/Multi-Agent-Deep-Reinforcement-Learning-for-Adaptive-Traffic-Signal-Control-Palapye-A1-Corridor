"""
vehicle_detector.py
====================
Extracts per-lane queue lengths from a live video feed using YOLOv8.
Produces the same halting-vehicle counts that SUMO's TraCI returns,
so the values can feed directly into the PPO observation vector.

Concept flow:
    Camera / video file
        → YOLOv8 vehicle detection
        → ROI (region-of-interest) per lane
        → stationary-vehicle filter (optical flow)
        → queue count per lane   ← same as traci.lane.getLastStepHaltingNumber()
        → log-scaled observation  ← ready for PPO model

Requirements:
    pip install ultralytics opencv-python numpy

Usage:
    detector = VehicleDetector(source=0)          # 0 = webcam, or path to video
    detector.define_roi("lane_north", [(x1,y1), (x2,y2), (x3,y3), (x4,y4)])
    detector.run()   # opens window; press Q to quit
"""

import cv2
import numpy as np
from ultralytics import YOLO
from collections import defaultdict, deque

# COCO class IDs for vehicles
VEHICLE_CLASSES = [2, 3, 5, 7]  # car, motorcycle, bus, truck

# How many pixels a vehicle's centre can move and still be considered stationary
STATIONARY_THRESHOLD_PX = 8

# How many consecutive frames a vehicle must be still to count as queued
STATIONARY_MIN_FRAMES = 5

# Match sumo_env.py constants so values feed directly into the PPO obs
OBS_MAX_QUEUE  = 15.0
LOG_OBS_MAX    = float(np.log1p(OBS_MAX_QUEUE))
OBS_MAX_WAIT   = 300.0
LOG_OBS_MAX_WAIT = float(np.log1p(OBS_MAX_WAIT))


class VehicleDetector:
    def __init__(self, source=0, model_path="yolov8n.pt", conf=0.4):
        """
        source     : camera index (0) or path to a video file
        model_path : YOLOv8 weights; 'yolov8n.pt' auto-downloads on first run
        conf       : detection confidence threshold
        """
        self.source = source
        self.conf   = conf
        self.model  = YOLO(model_path)

        # lane_id -> list of (x, y) polygon vertices defining the ROI
        self._rois: dict[str, list[tuple[int, int]]] = {}

        # track_id -> deque of (cx, cy) centres across recent frames
        self._history: dict[int, deque] = defaultdict(lambda: deque(maxlen=STATIONARY_MIN_FRAMES))

        # track_id -> frames spent stationary
        self._stationary_frames: dict[int, int] = defaultdict(int)

        # Results from last processed frame
        self.queue_counts: dict[str, int] = {}
        self.vehicle_boxes: list = []

    # ──────────────────────────────────────────────────────────────
    # ROI management
    # ──────────────────────────────────────────────────────────────
    def define_roi(self, lane_id: str, polygon: list[tuple[int, int]]):
        """
        Register a lane with its detection zone.
        polygon is a list of (x, y) pixel coordinates forming a closed polygon.
        Tip: use define_rois_interactive() below to draw them with your mouse.

        Example for a northbound approach lane:
            detector.define_roi("lane_N1", [(120,400), (200,400), (220,600), (100,600)])
        """
        self._rois[lane_id] = polygon

    def define_rois_interactive(self, frame: np.ndarray):
        """
        Opens an interactive window so you can click to define ROI polygons.
        Left-click to add points, right-click to finish a polygon, Q to quit.
        Prints the resulting define_roi() calls you can paste into your code.
        """
        clone    = frame.copy()
        points   = []
        lane_idx = [0]
        results  = {}

        def _mouse(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                points.append((x, y))
                cv2.circle(clone, (x, y), 4, (0, 255, 0), -1)
                if len(points) > 1:
                    cv2.line(clone, points[-2], points[-1], (0, 255, 0), 2)
                cv2.imshow("Define ROIs", clone)
            elif event == cv2.EVENT_RBUTTONDOWN and len(points) >= 3:
                lane_id = f"lane_{lane_idx[0]}"
                results[lane_id] = list(points)
                cv2.polylines(clone, [np.array(points)], True, (0, 200, 255), 2)
                cv2.imshow("Define ROIs", clone)
                print(f'detector.define_roi("{lane_id}", {list(points)})')
                points.clear()
                lane_idx[0] += 1

        cv2.namedWindow("Define ROIs")
        cv2.setMouseCallback("Define ROIs", _mouse)
        cv2.imshow("Define ROIs", clone)

        while True:
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cv2.destroyWindow("Define ROIs")
        for lid, poly in results.items():
            self.define_roi(lid, poly)

    # ──────────────────────────────────────────────────────────────
    # Core detection
    # ──────────────────────────────────────────────────────────────
    def _is_in_roi(self, cx: int, cy: int, polygon: list[tuple[int, int]]) -> bool:
        pt  = (float(cx), float(cy))
        pts = np.array(polygon, dtype=np.float32)
        return cv2.pointPolygonTest(pts, pt, False) >= 0

    def _update_tracker(self, track_id: int, cx: int, cy: int) -> bool:
        """
        Returns True if this vehicle has been stationary for STATIONARY_MIN_FRAMES.
        """
        history = self._history[track_id]
        history.append((cx, cy))

        if len(history) < STATIONARY_MIN_FRAMES:
            return False

        xs = [p[0] for p in history]
        ys = [p[1] for p in history]
        movement = max(max(xs) - min(xs), max(ys) - min(ys))
        return movement <= STATIONARY_THRESHOLD_PX

    def process_frame(self, frame: np.ndarray) -> dict[str, int]:
        """
        Run detection on a single frame.
        Returns {lane_id: halting_count} matching traci.lane.getLastStepHaltingNumber().
        """
        results = self.model.track(
            frame,
            classes=VEHICLE_CLASSES,
            conf=self.conf,
            persist=True,
            verbose=False,
        )

        boxes = results[0].boxes
        self.vehicle_boxes = boxes

        counts = {lane_id: 0 for lane_id in self._rois}

        if boxes is None or boxes.id is None:
            self.queue_counts = counts
            return counts

        for box, track_id in zip(boxes.xyxy, boxes.id.int()):
            x1, y1, x2, y2 = map(int, box)
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2
            tid = int(track_id)

            is_stationary = self._update_tracker(tid, cx, cy)

            if is_stationary:
                for lane_id, polygon in self._rois.items():
                    if self._is_in_roi(cx, cy, polygon):
                        counts[lane_id] += 1

        self.queue_counts = counts
        return counts

    # ──────────────────────────────────────────────────────────────
    # PPO observation helpers
    # ──────────────────────────────────────────────────────────────
    def to_ppo_obs(self, lane_order: list[str]) -> np.ndarray:
        """
        Converts current queue_counts into log-scaled values
        matching the observation format in sumo_env.py.

        lane_order: ordered list of lane IDs matching how the env indexes lanes.
        Returns a 1-D float32 array of length len(lane_order).
        """
        obs = []
        for lid in lane_order:
            h = self.queue_counts.get(lid, 0)
            obs.append(min(float(np.log1p(h) / LOG_OBS_MAX), 1.0))
        return np.array(obs, dtype=np.float32)

    # ──────────────────────────────────────────────────────────────
    # Visualisation
    # ──────────────────────────────────────────────────────────────
    def _draw_overlay(self, frame: np.ndarray) -> np.ndarray:
        out = frame.copy()

        # Draw ROI polygons
        for lane_id, polygon in self._rois.items():
            pts = np.array(polygon, dtype=np.int32)
            cv2.polylines(out, [pts], True, (0, 200, 255), 2)
            cx = int(np.mean([p[0] for p in polygon]))
            cy = int(np.mean([p[1] for p in polygon]))
            count = self.queue_counts.get(lane_id, 0)
            cv2.putText(out, f"{lane_id}: {count}", (cx - 40, cy),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)

        # Draw bounding boxes
        if self.vehicle_boxes is not None and self.vehicle_boxes.xyxy is not None:
            for box in self.vehicle_boxes.xyxy:
                x1, y1, x2, y2 = map(int, box)
                cv2.rectangle(out, (x1, y1), (x2, y2), (50, 200, 50), 2)

        # Queue summary panel
        y_off = 30
        cv2.rectangle(out, (10, 10), (260, 30 + 28 * len(self._rois)), (0, 0, 0), -1)
        for lane_id, count in self.queue_counts.items():
            cv2.putText(out, f"{lane_id:20s} : {count:2d}", (15, y_off),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
            y_off += 28

        return out

    # ──────────────────────────────────────────────────────────────
    # Main loop
    # ──────────────────────────────────────────────────────────────
    def run(self):
        """
        Opens video source, processes frames, and displays annotated output.
        Press Q to quit.  Press R on the first frame to define ROIs interactively.
        """
        cap = cv2.VideoCapture(self.source)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video source: {self.source}")

        print("[DETECTOR] Running — press Q to quit, R to define ROIs interactively.")

        first_frame = True
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            if key == ord('r') and first_frame:
                self.define_rois_interactive(frame)

            first_frame = False
            self.process_frame(frame)
            display = self._draw_overlay(frame)

            # Print queue counts to console each frame
            print(f"\rQueue: { {k: v for k, v in self.queue_counts.items()} }   ", end="")

            cv2.imshow("Vehicle Detector — Queue Extraction", display)

        cap.release()
        cv2.destroyAllWindows()
        print("\n[DETECTOR] Stopped.")


# ──────────────────────────────────────────────────────────────────
# Demo
# ──────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    detector = VehicleDetector(
        source=0,           # 0 = webcam; replace with "video.mp4" for a file
        model_path="yolov8n.pt",
        conf=0.4,
    )

    # Define ROIs for each lane approach (pixel coordinates for your camera view).
    # Replace these with your actual intersection geometry.
    # Or run the script and press R on the first frame to draw them interactively.
    detector.define_roi("lane_N1", [(100, 400), (200, 400), (210, 600), (90, 600)])
    detector.define_roi("lane_N2", [(210, 400), (310, 400), (320, 600), (200, 600)])
    detector.define_roi("lane_S1", [(350, 100), (450, 100), (460, 300), (340, 300)])

    detector.run()
