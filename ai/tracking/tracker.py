import time
from collections import deque
from typing import Dict, List, Tuple, Optional
from ai.detection.detector import DetectionResult

class TrackHistoryManager:
    """Maintains trajectory history and temporal stats for tracked vehicles and persons."""
    def __init__(self, max_history_len: int = 30, max_inactive_frames: int = 60):
        self.max_history_len = max_history_len
        self.max_inactive_frames = max_inactive_frames
        # Map track_id -> deque of (frame_idx, (x, y))
        self.tracks: Dict[int, deque] = {}
        # Map track_id -> last seen frame index
        self.last_seen: Dict[int, int] = {}
        # Map track_id -> class_name
        self.track_classes: Dict[int, str] = {}
        self.current_frame: int = 0

    def update(self, detections: List[DetectionResult], frame_idx: int):
        self.current_frame = frame_idx
        for det in detections:
            if det.track_id is None:
                continue
            t_id = det.track_id
            if t_id not in self.tracks:
                self.tracks[t_id] = deque(maxlen=self.max_history_len)
            
            # Bottom center is usually the contact point on the road
            self.tracks[t_id].append((frame_idx, det.bottom_center))
            self.last_seen[t_id] = frame_idx
            self.track_classes[t_id] = det.class_name

        # Clean inactive tracks periodically
        if frame_idx % 30 == 0:
            self._prune_inactive()

    def _prune_inactive(self):
        to_del = []
        for t_id, last_f in self.last_seen.items():
            if self.current_frame - last_f > self.max_inactive_frames:
                to_del.append(t_id)
        for t_id in to_del:
            self.tracks.pop(t_id, None)
            self.last_seen.pop(t_id, None)
            self.track_classes.pop(t_id, None)

    def get_trajectory(self, track_id: int) -> List[Tuple[float, float]]:
        if track_id not in self.tracks:
            return []
        return [pt for _, pt in self.tracks[track_id]]

    def get_last_positions(self, track_id: int, count: int = 2) -> List[Tuple[float, float]]:
        """Returns the most recent `count` positions for the track."""
        if track_id not in self.tracks or len(self.tracks[track_id]) < 2:
            return []
        pts = [pt for _, pt in self.tracks[track_id]]
        return pts[-count:]
