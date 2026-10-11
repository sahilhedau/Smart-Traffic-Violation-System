import logging
from typing import List, Dict, Any, Optional
import torch
import numpy as np

logger = logging.getLogger("traffic_system.detector")

# COCO Class mapping relevant to traffic violation detection
# 0: person, 1: bicycle, 2: car, 3: motorcycle, 5: bus, 7: truck, 9: traffic light
TRAFFIC_CLASSES = {
    0: "person",
    1: "bicycle",
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck",
    9: "traffic light"
}

class DetectionResult:
    def __init__(
        self,
        box: List[float],  # [x1, y1, x2, y2] in pixels
        class_id: int,
        class_name: str,
        confidence: float,
        track_id: Optional[int] = None
    ):
        self.box = box
        self.class_id = class_id
        self.class_name = class_name
        self.confidence = confidence
        self.track_id = track_id

    @property
    def x1(self) -> float:
        return self.box[0]

    @property
    def y1(self) -> float:
        return self.box[1]

    @property
    def x2(self) -> float:
        return self.box[2]

    @property
    def y2(self) -> float:
        return self.box[3]

    @property
    def width(self) -> float:
        return max(0.0, self.box[2] - self.box[0])

    @property
    def height(self) -> float:
        return max(0.0, self.box[3] - self.box[1])

    @property
    def center(self) -> tuple[float, float]:
        return ((self.box[0] + self.box[2]) / 2.0, (self.box[1] + self.box[3]) / 2.0)

    @property
    def bottom_center(self) -> tuple[float, float]:
        return ((self.box[0] + self.box[2]) / 2.0, self.box[3])

    def to_dict(self) -> Dict[str, Any]:
        return {
            "box": [round(c, 2) for c in self.box],
            "class_id": self.class_id,
            "class_name": self.class_name,
            "confidence": round(self.confidence, 3),
            "track_id": self.track_id
        }

class ObjectDetector:
    def __init__(self, model_name: str = "yolov8n.pt", device: str = "auto", conf_threshold: float = 0.35):
        self.model_name = model_name
        self.conf_threshold = conf_threshold
        self.device = self._resolve_device(device)
        self.model = None
        self._load_model()

    def _resolve_device(self, requested: str) -> str:
        if requested == "auto":
            if torch.cuda.is_available():
                device_name = torch.cuda.get_device_name(0)
                logger.info("NVIDIA CUDA detected (%s). Using GPU inference.", device_name)
                return "cuda"
            logger.info("No CUDA GPU detected or available. Using CPU inference.")
            return "cpu"
        return requested

    def _load_model(self):
        try:
            from ultralytics import YOLO
            logger.info("Loading YOLO model '%s' on %s...", self.model_name, self.device)
            self.model = YOLO(self.model_name)
            logger.info("YOLO model loaded successfully.")
        except Exception as e:
            logger.error("Failed to load YOLO model %s: %s", self.model_name, e)
            self.model = None

    def detect_and_track(self, frame: np.ndarray, persist: bool = True) -> List[DetectionResult]:
        """Runs detection and ByteTrack tracking on a single frame."""
        if self.model is None:
            return []

        try:
            # Classes filter: persons, bicycles, cars, motorcycles, buses, trucks, traffic lights
            classes_to_track = list(TRAFFIC_CLASSES.keys())
            results = self.model.track(
                source=frame,
                persist=persist,
                conf=self.conf_threshold,
                classes=classes_to_track,
                device=self.device,
                verbose=False,
                tracker="bytetrack.yaml"
            )

            detections: List[DetectionResult] = []
            if not results or len(results) == 0:
                return detections

            r = results[0]
            if r.boxes is None or len(r.boxes) == 0:
                return detections

            boxes_xyxy = r.boxes.xyxy.cpu().numpy()
            confs = r.boxes.conf.cpu().numpy()
            class_ids = r.boxes.cls.cpu().numpy().astype(int)
            track_ids = r.boxes.id.cpu().numpy().astype(int) if r.boxes.id is not None else [None] * len(boxes_xyxy)

            for box, conf, cls_id, trk_id in zip(boxes_xyxy, confs, class_ids, track_ids):
                class_name = TRAFFIC_CLASSES.get(cls_id, self.model.names.get(cls_id, f"cls_{cls_id}"))
                detections.append(DetectionResult(
                    box=[float(box[0]), float(box[1]), float(box[2]), float(box[3])],
                    class_id=int(cls_id),
                    class_name=class_name,
                    confidence=float(conf),
                    track_id=int(trk_id) if trk_id is not None else None
                ))

            return detections
        except Exception as e:
            logger.error("Error during object detection & tracking: %s", e)
            return []
