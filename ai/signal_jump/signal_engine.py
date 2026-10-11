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
        stop_line: List[float],    # [x1, y1, x2, y2] normalized (0.0 - 1.0) or absolute
        signal_roi: List[float],   # [x1, y1, x2, y2] normalized
        signal_mode: str = "red",  # 'auto_detect', 'red', 'yellow', 'green'
    ):
        self.stop_line_norm = stop_line
        self.signal_roi_norm = signal_roi
        self.signal_mode = signal_mode
        self.alerted_tracks: set[int] = set()

    def determine_signal_state(self, frame: np.ndarray) -> Tuple[str, float]:
        """
        Determines traffic light state.
        Returns: (state, confidence) where state in ['RED', 'YELLOW', 'GREEN', 'UNKNOWN']
        """
        if self.signal_mode in ["red", "yellow", "green"]:
            # Calibrated test/demo state override
            return self.signal_mode.upper(), 1.0

        h, w = frame.shape[:2]
        rx1 = int(self.signal_roi_norm[0] * w)
        ry1 = int(self.signal_roi_norm[1] * h)
        rx2 = int(self.signal_roi_norm[2] * w)
        ry2 = int(self.signal_roi_norm[3] * h)

        rx1, ry1 = max(0, rx1), max(0, ry1)
        rx2, ry2 = min(w, rx2), min(h, ry2)

        if rx2 <= rx1 or ry2 <= ry1:
            return "UNKNOWN", 0.0

        roi = frame[ry1:ry2, rx1:rx2]
        if roi.size == 0:
            return "UNKNOWN", 0.0

        # Convert to HSV to check dominant signal color
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

        # Red masks (wraps around 0 and 180)
        mask_red1 = cv2.inRange(hsv, np.array([0, 100, 100]), np.array([10, 255, 255]))
        mask_red2 = cv2.inRange(hsv, np.array([160, 100, 100]), np.array([180, 255, 255]))
        mask_red = cv2.bitwise_or(mask_red1, mask_red2)

        # Yellow mask
        mask_yellow = cv2.inRange(hsv, np.array([15, 100, 100]), np.array([35, 255, 255]))

        # Green mask
        mask_green = cv2.inRange(hsv, np.array([40, 100, 100]), np.array([85, 255, 255]))

        red_pixels = int(cv2.countNonZero(mask_red))
        yellow_pixels = int(cv2.countNonZero(mask_yellow))
        green_pixels = int(cv2.countNonZero(mask_green))

        total_colored = red_pixels + yellow_pixels + green_pixels
        min_activation = max(20, int(roi.shape[0] * roi.shape[1] * 0.02))

        if total_colored < min_activation:
            # Not enough colored pixels detected in the ROI
            return "UNKNOWN", 0.0

        if red_pixels > yellow_pixels and red_pixels > green_pixels:
            conf = red_pixels / total_colored
            return "RED", float(conf)
        elif green_pixels > red_pixels and green_pixels > yellow_pixels:
            conf = green_pixels / total_colored
            return "GREEN", float(conf)
        elif yellow_pixels > red_pixels and yellow_pixels > green_pixels:
            conf = yellow_pixels / total_colored
            return "YELLOW", float(conf)

        return "UNKNOWN", 0.0

    def check_violations(
        self,
        frame: np.ndarray,
        tracked_vehicles: List[DetectionResult],
        track_trajectories: Dict[int, List[Tuple[float, float]]]
    ) -> List[Dict[str, Any]]:
        """
        Checks if any tracked vehicle crossed the stop line while the signal is RED.
        Returns list of candidate violation event dictionaries.
        """
        violations = []
        signal_state, signal_conf = self.determine_signal_state(frame)

        # Violation ONLY occurs if signal is RED
        if signal_state != "RED":
            return violations

        h, w = frame.shape[:2]
        # Stop line in pixel coordinates: A=(lx1, ly1), B=(lx2, ly2)
        A = (self.stop_line_norm[0] * w, self.stop_line_norm[1] * h)
        B = (self.stop_line_norm[2] * w, self.stop_line_norm[3] * h)

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

            # Did the trajectory cross the stop line?
            if segments_intersect(A, B, prev_pos, curr_pos):
                # Verify crossing direction (moving forward across the line)
                # Determinant sign indicates side:
                c1 = ccw(A, B, prev_pos)
                c2 = ccw(A, B, curr_pos)
                if c1 < 0 and c2 > 0:  # standard crossing direction
                    self.alerted_tracks.add(det.track_id)
                    violations.append({
                        "event_type": "RED_LIGHT_CROSSING",
                        "tracking_id": det.track_id,
                        "class_name": det.class_name,
                        "confidence": det.confidence,
                        "box": det.box,
                        "signal_state": signal_state,
                        "signal_confidence": signal_conf,
                        "crossing_pos": curr_pos,
                        "severity": "HIGH",
                    })

        return violations

    def draw_overlay(self, frame: np.ndarray, signal_state: str):
        """Draws stop line and traffic light indicator on the video frame."""
        h, w = frame.shape[:2]
        lx1, ly1 = int(self.stop_line_norm[0] * w), int(self.stop_line_norm[1] * h)
        lx2, ly2 = int(self.stop_line_norm[2] * w), int(self.stop_line_norm[3] * h)

        # Draw stop line
        line_color = (0, 0, 255) if signal_state == "RED" else (0, 255, 0)
        cv2.line(frame, (lx1, ly1), (lx2, ly2), line_color, 3)
        cv2.putText(
            frame, "STOP LINE", (lx1 + 10, max(20, ly1 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, line_color, 2
        )

        # Draw Signal Box / Status indicator in top right
        rx1, ry1 = int(self.signal_roi_norm[0] * w), int(self.signal_roi_norm[1] * h)
        rx2, ry2 = int(self.signal_roi_norm[2] * w), int(self.signal_roi_norm[3] * h)
        cv2.rectangle(frame, (rx1, ry1), (rx2, ry2), (255, 255, 255), 2)
        cv2.putText(
            frame, f"LIGHT: {signal_state}", (rx1, max(20, ry1 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, line_color, 2
        )
