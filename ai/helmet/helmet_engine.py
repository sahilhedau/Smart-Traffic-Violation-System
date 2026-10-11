import cv2
import numpy as np
import logging
from pathlib import Path
from collections import defaultdict, deque
from typing import List, Dict, Any, Tuple, Optional
from ai.detection.detector import DetectionResult

logger = logging.getLogger("traffic_system.helmet")

class HelmetDetector:
    def __init__(
        self,
        model_path: Optional[str] = None,
        min_confidence: float = 0.40,
        smoothing_frames: int = 3,
        history_len: int = 10
    ):
        self.min_confidence = min_confidence
        self.smoothing_frames = smoothing_frames
        self.history_len = history_len
        self.specialized_model = None
        self.alerted_tracks: set[int] = set()
        # Map rider_track_id or moto_track_id -> deque of classifications: ['HELMET', 'NO_HELMET', 'UNKNOWN']
        self.decision_history: Dict[int, deque] = defaultdict(lambda: deque(maxlen=self.history_len))

        if model_path and Path(model_path).exists():
            try:
                from ultralytics import YOLO
                logger.info("Loading specialized helmet detection model from %s...", model_path)
                self.specialized_model = YOLO(model_path)
                logger.info("Specialized helmet model loaded.")
            except Exception as e:
                logger.warning("Could not load specialized helmet model: %s. Using computer-vision baseline.", e)

    def extract_head_crop(self, frame: np.ndarray, person_box: List[float]) -> Optional[np.ndarray]:
        """Extracts upper 25% of the person's bounding box as the head region."""
        h, w = frame.shape[:2]
        px1, py1, px2, py2 = person_box
        pw = px2 - px1
        ph = py2 - py1

        head_y1 = max(0, int(py1))
        head_y2 = min(h, int(py1 + 0.25 * ph))
        head_x1 = max(0, int(px1))
        head_x2 = min(w, int(px2))

        if head_y2 <= head_y1 or head_x2 <= head_x1:
            return None

        crop = frame[head_y1:head_y2, head_x1:head_x2]
        if crop.shape[0] < 20 or crop.shape[1] < 20:
            # Low resolution: uncertain/unknown
            return None

        return crop

    def classify_head_baseline(self, head_crop: np.ndarray) -> Tuple[str, float]:
        """
        Baseline computer-vision classifier for head region when specialized model is not provided.
        Returns: ('HELMET' | 'NO_HELMET' | 'UNKNOWN', confidence)
        """
        if head_crop is None or head_crop.size == 0:
            return "UNKNOWN", 0.0

        h, w = head_crop.shape[:2]
        if h < 20 or w < 20:
            return "UNKNOWN", 0.0

        hsv = cv2.cvtColor(head_crop, cv2.COLOR_BGR2HSV)
        gray = cv2.cvtColor(head_crop, cv2.COLOR_BGR2GRAY)

        # 1. Skin tone detection (HSV)
        # Typical skin range in HSV
        lower_skin = np.array([0, 30, 60], dtype=np.uint8)
        upper_skin = np.array([25, 170, 255], dtype=np.uint8)
        skin_mask = cv2.inRange(hsv, lower_skin, upper_skin)
        skin_pixels = cv2.countNonZero(skin_mask)
        total_pixels = h * w
        skin_ratio = skin_pixels / float(total_pixels)

        # 2. Edge / texture density (hair/face vs smooth helmet shell)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()

        # 3. Helmet shell color uniformity & saturation
        # Helmets typically have high saturation colors (white, red, blue, black gloss) or very uniform hue
        saturation = hsv[:, :, 1]
        mean_sat = np.mean(saturation)

        # Decision rule:
        # High skin ratio in head crop + high texture variation indicates bare head/face without helmet
        if skin_ratio > 0.28 and laplacian_var > 60:
            conf = min(0.90, 0.45 + skin_ratio)
            return "NO_HELMET", float(conf)
        elif skin_ratio < 0.12 and (mean_sat > 70 or laplacian_var < 45):
            conf = 0.70
            return "HELMET", float(conf)

        return "UNKNOWN", 0.35

    def classify_rider(self, frame: np.ndarray, person_box: List[float]) -> Tuple[str, float]:
        """Classifies if rider has a helmet using specialized model or baseline."""
        if self.specialized_model is not None:
            head_crop = self.extract_head_crop(frame, person_box)
            if head_crop is None:
                return "UNKNOWN", 0.0
            try:
                results = self.specialized_model.predict(head_crop, conf=self.min_confidence, verbose=False)
                if results and len(results) > 0 and len(results[0].boxes) > 0:
                    cls_id = int(results[0].boxes.cls[0].item())
                    conf = float(results[0].boxes.conf[0].item())
                    cls_name = results[0].names.get(cls_id, "").lower()
                    if "no" in cls_name or "without" in cls_name:
                        return "NO_HELMET", conf
                    elif "helmet" in cls_name:
                        return "HELMET", conf
            except Exception as e:
                logger.error("Inference error in specialized helmet model: %s", e)

        # Fallback to CV baseline
        head_crop = self.extract_head_crop(frame, person_box)
        return self.classify_head_baseline(head_crop)

    def process_riders(
        self,
        frame: np.ndarray,
        moto_riders_map: Dict[int, List[DetectionResult]],
        motorcycles: List[DetectionResult]
    ) -> List[Dict[str, Any]]:
        """
        Evaluates helmet usage for associated riders on each motorcycle.
        Returns list of candidate NO_HELMET violations.
        """
        violations = []

        moto_by_id = {m.track_id: m for m in motorcycles if m.track_id is not None}

        for m_id, riders in moto_riders_map.items():
            if m_id in self.alerted_tracks:
                continue

            moto = moto_by_id.get(m_id)
            if not moto:
                continue

            for rider in riders:
                pred, conf = self.classify_rider(frame, rider.box)
                self.decision_history[m_id].append(pred)

                # Temporal smoothing
                recent = list(self.decision_history[m_id])
                if len(recent) >= self.smoothing_frames:
                    no_helmet_count = sum(1 for d in recent[-self.smoothing_frames:] if d == "NO_HELMET")
                    if no_helmet_count >= (self.smoothing_frames - 1) and conf >= self.min_confidence:
                        self.alerted_tracks.add(m_id)
                        violations.append({
                            "event_type": "NO_HELMET",
                            "tracking_id": m_id,
                            "rider_box": rider.box,
                            "moto_box": moto.box,
                            "confidence": round(conf, 3),
                            "severity": "HIGH",
                            "notes": f"Rider without helmet detected on motorcycle #{m_id} (confidence: {conf:.2f})"
                        })
                        break  # Only one alert per motorcycle track

        return violations
