import logging
from collections import defaultdict, deque
from typing import List, Dict, Any, Tuple, Optional
from ai.detection.detector import DetectionResult

logger = logging.getLogger("traffic_system.triple_riding")

def calculate_iou(boxA: List[float], boxB: List[float]) -> float:
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interArea = max(0.0, xB - xA) * max(0.0, yB - yA)
    boxAArea = max(0.0, boxA[2] - boxA[0]) * max(0.0, boxA[3] - boxA[1])
    boxBArea = max(0.0, boxB[2] - boxB[0]) * max(0.0, boxB[3] - boxB[1])

    iou = interArea / float(boxAArea + boxBArea - interArea + 1e-6)
    return iou

class TripleRidingDetector:
    def __init__(
        self,
        min_riders: int = 3,
        min_frames: int = 3,
        history_len: int = 10
    ):
        self.min_riders = min_riders
        self.min_frames = min_frames
        self.history_len = history_len
        # Map motorcycle_track_id -> deque of rider counts
        self.rider_history: Dict[int, deque] = defaultdict(lambda: deque(maxlen=self.history_len))
        self.alerted_tracks: set[int] = set()

    def is_rider_on_motorcycle(self, person: DetectionResult, moto: DetectionResult) -> bool:
        """Determines if a person detection is seated on/riding the motorcycle."""
        px1, py1, px2, py2 = person.box
        mx1, my1, mx2, my2 = moto.box
        mw = moto.width
        mh = moto.height

        # Rider zone: extends above motorcycle to include riders' torsos and heads
        rider_zone = [
            mx1 - 0.2 * mw,
            max(0.0, my1 - 0.9 * mh),
            mx2 + 0.2 * mw,
            my2
        ]

        # Calculate overlap of person with the rider zone
        inter_x1 = max(px1, rider_zone[0])
        inter_y1 = max(py1, rider_zone[1])
        inter_x2 = min(px2, rider_zone[2])
        inter_y2 = min(py2, rider_zone[3])

        inter_area = max(0.0, inter_x2 - inter_x1) * max(0.0, inter_y2 - inter_y1)
        person_area = person.width * person.height

        if person_area <= 0:
            return False

        overlap_ratio = inter_area / person_area

        # Person's feet/bottom should be inside or near motorcycle height
        person_bottom = py2
        vertical_valid = (my1 - 0.3 * mh) <= person_bottom <= (my2 + 0.1 * mh)

        # Person's horizontal center should be near motorcycle
        pcx = (px1 + px2) / 2.0
        horizontal_valid = (mx1 - 0.3 * mw) <= pcx <= (mx2 + 0.3 * mw)

        return overlap_ratio >= 0.45 and vertical_valid and horizontal_valid

    def associate_riders(
        self,
        motorcycles: List[DetectionResult],
        persons: List[DetectionResult]
    ) -> Dict[Any, List[DetectionResult]]:
        """Maps motorcycle index/track_id to list of associated rider DetectionResults."""
        mapping: Dict[Any, List[DetectionResult]] = {}
        for idx, moto in enumerate(motorcycles):
            key = moto.track_id if moto.track_id is not None else idx
            mapping[key] = [p for p in persons if self.is_rider_on_motorcycle(p, moto)]
        return mapping

    def process_frame(
        self,
        motorcycles: List[DetectionResult],
        persons: List[DetectionResult]
    ) -> Tuple[List[Dict[str, Any]], Dict[int, int]]:
        """
        Associates persons with motorcycles, tracks rider counts temporally,
        and generates candidate alerts for triple riding.
        Returns: (violations, rider_counts_by_track_id)
        """
        violations = []
        counts_map: Dict[int, int] = {}

        for moto in motorcycles:
            m_id = moto.track_id
            associated_riders = []

            for person in persons:
                if self.is_rider_on_motorcycle(person, moto):
                    associated_riders.append(person)

            rider_count = len(associated_riders)
            if m_id is not None:
                counts_map[m_id] = rider_count
                self.rider_history[m_id].append(rider_count)

                # Temporal smoothing: check if last N frames consistently showed >= 3 riders
                recent_counts = list(self.rider_history[m_id])
                if len(recent_counts) >= self.min_frames:
                    # If majority of recent frames >= min_riders
                    high_rider_frames = sum(1 for c in recent_counts[-self.min_frames:] if c >= self.min_riders)
                    if high_rider_frames >= (self.min_frames - 1) and m_id not in self.alerted_tracks:
                        self.alerted_tracks.add(m_id)
                        violations.append({
                            "event_type": "TRIPLE_RIDING",
                            "tracking_id": m_id,
                            "rider_count": max(recent_counts),
                            "confidence": moto.confidence,
                            "box": moto.box,
                            "rider_boxes": [r.box for r in associated_riders],
                            "severity": "HIGH",
                            "notes": f"Detected {rider_count} riders on motorcycle track #{m_id}"
                        })

        return violations, counts_map
