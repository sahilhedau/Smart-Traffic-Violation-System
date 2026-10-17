import cv2
import numpy as np
import logging
from typing import List, Tuple, Optional, Dict, Any
from ai.detection.detector import DetectionResult

logger = logging.getLogger("traffic_system.signal_jump")

def ccw(A: Tuple[float, float], B: Tuple[float, float], C: Tuple[float, float]) -> float:
    """Returns orientation determinant of triplet (A, B, C)."""
    return (B[0] - A[0]) * (C[1] - A[1]) - (B[1] - A[1]) * (C[0] - A[0])

def segments_intersect(
    A: Tuple[float, float], B: Tuple[float, float],
    C: Tuple[float, float], D: Tuple[float, float]
) -> bool:
    """Checks if line segment AB and line segment CD strictly intersect."""
    ccw_ab_c = ccw(A, B, C)
    ccw_ab_d = ccw(A, B, D)
    ccw_cd_a = ccw(C, D, A)
    ccw_cd_b = ccw(C, D, B)

    return (ccw_ab_c * ccw_ab_d < 0) and (ccw_cd_a * ccw_cd_b < 0)

class SignalJumpDetector:
    def __init__(
        self,
        stop_line: List[float],    # [x1, y1, x2, y2] normalized (0.0 - 1.0)
        signal_roi: List[float],   # [x1, y1, x2, y2] normalized
        signal_mode: str = "auto_detect",  # 'auto_detect', 'red', 'yellow', 'green'
        signal_rois: Optional[List[List[float]]] = None,
        stop_lines: Optional[List[List[float]]] = None,
    ):
        self.signal_rois_norm = signal_rois or [signal_roi]
        self.stop_lines_norm = stop_lines or [stop_line] * len(self.signal_rois_norm)
        if len(self.stop_lines_norm) != len(self.signal_rois_norm):
            raise ValueError("Each signal region must have a corresponding stop line")
        self.signal_mode = signal_mode
        self.alerted_tracks: set[int] = set()

    def determine_signal_state(
        self,
        frame: np.ndarray,
        traffic_light_detections: Optional[List[DetectionResult]] = None
    ) -> Tuple[str, float]:
        """
        Determines traffic light state.
        Returns: (state, confidence) where state in ['RED', 'YELLOW', 'GREEN', 'UNKNOWN']
        """
        states = self.determine_signal_states(frame, traffic_light_detections)
        detected = [state for state in states if state[0] != "UNKNOWN"]
        return max(detected, key=lambda item: item[1]) if detected else ("UNKNOWN", 0.0)

    def determine_signal_states(
        self,
        frame: np.ndarray,
        traffic_light_detections: Optional[List[DetectionResult]] = None
    ) -> List[Tuple[str, float]]:
        """Return one classified signal state for each calibrated signal region."""
        if self.signal_mode in ["red", "yellow", "green"]:
            return [(self.signal_mode.upper(), 1.0)] * len(self.signal_rois_norm)

        h, w = frame.shape[:2]
        roi_boxes = [
            self._normalized_box_to_pixels(roi, w, h)
            for roi in self.signal_rois_norm
        ]
        light_boxes = []
        if traffic_light_detections:
            light_boxes = [
                self._clip_box(detection.box, w, h)
                for detection in traffic_light_detections
                if detection.class_name == "traffic light"
            ]
            light_boxes = [box for box in light_boxes if box is not None]

        states = []
        for rx1, ry1, rx2, ry2 in roi_boxes:
            if rx2 <= rx1 or ry2 <= ry1:
                states.append(("UNKNOWN", 0.0))
                continue
            regions = []
            for bx1, by1, bx2, by2 in light_boxes:
                ix1, iy1 = max(rx1, bx1), max(ry1, by1)
                ix2, iy2 = min(rx2, bx2), min(ry2, by2)
                if ix2 > ix1 and iy2 > iy1:
                    regions.append(frame[iy1:iy2, ix1:ix2])
            if not regions:
                regions = [frame[ry1:ry2, rx1:rx2]]
            detected = [
                state for state in map(self._classify_signal_region, regions)
                if state[0] != "UNKNOWN"
            ]
            states.append(max(detected, key=lambda item: item[1]) if detected else ("UNKNOWN", 0.0))
        return states

    @staticmethod
    def _normalized_box_to_pixels(box: List[float], width: int, height: int) -> Tuple[int, int, int, int]:
        x1, y1, x2, y2 = box
        return (
            max(0, min(width, int(x1 * width))),
            max(0, min(height, int(y1 * height))),
            max(0, min(width, int(x2 * width))),
            max(0, min(height, int(y2 * height))),
        )

    @staticmethod
    def _clip_box(box: List[float], width: int, height: int) -> Optional[Tuple[int, int, int, int]]:
        x1, y1, x2, y2 = map(int, box)
        clipped = (
            max(0, min(width, x1)),
            max(0, min(height, y1)),
            max(0, min(width, x2)),
            max(0, min(height, y2)),
        )
        return clipped if clipped[2] > clipped[0] and clipped[3] > clipped[1] else None

    @staticmethod
    def _classify_signal_region(region: np.ndarray) -> Tuple[str, float]:
        if region is None or region.size == 0:
            return "UNKNOWN", 0.0

        hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
        masks = {
            "RED": cv2.bitwise_or(
                cv2.inRange(hsv, np.array([0, 80, 80]), np.array([12, 255, 255])),
                cv2.inRange(hsv, np.array([168, 80, 80]), np.array([180, 255, 255]))
            ),
            "YELLOW": cv2.inRange(hsv, np.array([15, 80, 80]), np.array([38, 255, 255])),
            "GREEN": cv2.inRange(hsv, np.array([35, 70, 80]), np.array([95, 255, 255])),
        }
        counts = {state: int(cv2.countNonZero(mask)) for state, mask in masks.items()}
        colored_pixels = sum(counts.values())
        minimum_pixels = max(3, int(region.shape[0] * region.shape[1] * 0.01))
        if colored_pixels < minimum_pixels:
            return "UNKNOWN", 0.0

        state, pixels = max(counts.items(), key=lambda item: item[1])
        confidence = pixels / colored_pixels
        if pixels < minimum_pixels or confidence < 0.60:
            return "UNKNOWN", 0.0
        return state, float(confidence)

    def check_violations(
        self,
        frame: np.ndarray,
        tracked_vehicles: List[DetectionResult],
        track_trajectories: Dict[int, List[Tuple[float, float]]],
        traffic_light_detections: Optional[List[DetectionResult]] = None,
        signal_state: Optional[str] = None,
        signal_confidence: Optional[float] = None,
        signal_states: Optional[List[Tuple[str, float]]] = None
    ) -> List[Dict[str, Any]]:
        """
        Checks if any tracked vehicle crossed the stop line while the signal is RED.
        Returns list of candidate violation event dictionaries.
        """
        violations = []
        if signal_states is None:
            if signal_state is None:
                signal_states = self.determine_signal_states(frame, traffic_light_detections)
            else:
                confidence = signal_confidence if signal_confidence is not None else 1.0
                signal_states = [(signal_state, confidence)] * len(self.stop_lines_norm)

        h, w = frame.shape[:2]
        lines = [
            ((line[0] * w, line[1] * h), (line[2] * w, line[3] * h))
            for line in self.stop_lines_norm
        ]

        for det in tracked_vehicles:
            if det.track_id is None:
                continue
            if det.track_id in self.alerted_tracks:
                continue

            traj = track_trajectories.get(det.track_id, [])
            if len(traj) < 2:
                continue

            # Check segment from previous position to current position
            prev_pos = traj[-2]
            curr_pos = traj[-1]

            for line_index, (A, B) in enumerate(lines):
                if line_index >= len(signal_states):
                    break
                state, confidence = signal_states[line_index]
                if state != "RED":
                    continue
                if segments_intersect(A, B, prev_pos, curr_pos):
                    c1 = ccw(A, B, prev_pos)
                    c2 = ccw(A, B, curr_pos)
                    if c1 * c2 >= 0:
                        continue
                    self.alerted_tracks.add(det.track_id)
                    violations.append({
                        "event_type": "RED_LIGHT_CROSSING",
                        "tracking_id": det.track_id,
                        "class_name": det.class_name,
                        "confidence": det.confidence,
                        "box": det.box,
                        "signal_state": state,
                        "signal_confidence": confidence,
                        "signal_index": line_index,
                        "stop_line_index": line_index,
                        "crossing_pos": curr_pos,
                        "severity": "HIGH",
                    })
                    break

        return violations

    def draw_overlay(
        self,
        frame: np.ndarray,
        signal_state: str | List[Tuple[str, float]],
        traffic_light_detections: Optional[List[DetectionResult]] = None
    ):
        """Draws stop line and traffic light indicator on the video frame."""
        h, w = frame.shape[:2]
        states = signal_state if isinstance(signal_state, list) else [
            (signal_state, 1.0)
        ] * len(self.signal_rois_norm)
        colors = {
            "RED": (0, 0, 255),
            "YELLOW": (0, 200, 255),
            "GREEN": (0, 255, 0),
        }
        for index, line in enumerate(self.stop_lines_norm):
            state = states[index][0] if index < len(states) else "UNKNOWN"
            color = colors.get(state, (180, 180, 180))
            A = (int(line[0] * w), int(line[1] * h))
            B = (int(line[2] * w), int(line[3] * h))
            cv2.line(frame, A, B, color, 3)
            cv2.putText(
                frame, f"STOP {index + 1}: {state}", (A[0] + 10, max(20, A[1] - 10)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2
            )

        for index, roi in enumerate(self.signal_rois_norm):
            rx1, ry1, rx2, ry2 = self._normalized_box_to_pixels(roi, w, h)
            state = states[index][0] if index < len(states) else "UNKNOWN"
            color = colors.get(state, (180, 180, 180))
            cv2.rectangle(frame, (rx1, ry1), (rx2, ry2), (255, 255, 255), 2)
            cv2.putText(
                frame, f"LIGHT {index + 1}: {state}", (rx1, max(20, ry1 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2
            )
        for detection in traffic_light_detections or []:
            if detection.class_name != "traffic light":
                continue
            box = self._clip_box(detection.box, w, h)
            if box is not None:
                bx1, by1, bx2, by2 = box
                detection_color = (180, 180, 180)
                for index, roi in enumerate(self.signal_rois_norm):
                    rx1, ry1, rx2, ry2 = self._normalized_box_to_pixels(roi, w, h)
                    if min(rx2, bx2) > max(rx1, bx1) and min(ry2, by2) > max(ry1, by1):
                        state = states[index][0] if index < len(states) else "UNKNOWN"
                        detection_color = colors.get(state, detection_color)
                        break
                cv2.rectangle(frame, (bx1, by1), (bx2, by2), detection_color, 2)
