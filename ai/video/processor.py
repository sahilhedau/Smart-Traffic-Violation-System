import cv2
import time
import os
import uuid
import shutil
import subprocess
import logging
import numpy as np
from pathlib import Path
from datetime import datetime, timezone
from typing import Optional, Callable, Dict, Any, List

from backend.app.config import config
from backend.app.services.event_service import EventService
from backend.app.services.settings_service import SettingsService
from backend.app.schemas.event import ViolationEventCreate, EventType, Severity, ReviewStatus
from backend.app.database.db import get_db_connection

from ai.detection.detector import ObjectDetector, DetectionResult
from ai.tracking.tracker import TrackHistoryManager
from ai.signal_jump.signal_engine import SignalJumpDetector
from ai.triple_riding.triple_riding_engine import TripleRidingDetector
from ai.helmet.helmet_engine import HelmetDetector
from ai.license_plate.plate_engine import PlateRecognitionEngine

logger = logging.getLogger("traffic_system.video_processor")

class VideoProcessor:
    def __init__(
        self,
        job_id: str,
        video_source: str,
        output_path: Optional[str] = None,
        progress_callback: Optional[Callable[[int, int, float, int], None]] = None,
        is_demo: bool = False
    ):
        self.job_id = job_id
        self.video_source = str(video_source)
        self.output_path = output_path
        self.progress_callback = progress_callback
        self.is_demo = is_demo
        self.settings = SettingsService.get_settings()
        
        # Track active violations per tracking ID to display persistent annotations
        self.active_violations: Dict[int, List[str]] = {}
        self.recognized_plates: Dict[int, str] = {}
        self.tracked_plate_boxes: Dict[int, List[float]] = {}

    def process(self) -> Dict[str, Any]:
        """Runs end-to-end video processing and violation detection."""
        logger.info("Starting processing job %s for source: %s (is_demo: %s)", self.job_id, self.video_source, self.is_demo)
        
        cap = cv2.VideoCapture(self.video_source)
        if not cap.isOpened():
            err = f"Cannot open source video: {self.video_source}. File may be missing or format unsupported."
            logger.error(err)
            self._update_job_status("FAILED", error_message=err)
            raise ValueError(err)

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        orig_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        if total_frames <= 0:
            total_frames = 100
        if width <= 0 or height <= 0:
            width, height = 1280, 720
        if orig_fps <= 0 or orig_fps > 120:
            orig_fps = 25.0

        # Unique output path per job
        if not self.output_path:
            out_filename = f"annotated_{self.job_id}_{Path(self.video_source).stem}.mp4"
            self.output_path = str(config.ANNOTATED_VIDEOS_DIR / out_filename)

        # Temporary raw file for OpenCV writing before H.264 muxing
        raw_output_path = str(config.ANNOTATED_VIDEOS_DIR / f"temp_raw_{self.job_id}.mp4")

        # Select compatible fourcc
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(raw_output_path, fourcc, orig_fps, (width, height))
        if not writer.isOpened():
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            writer = cv2.VideoWriter(raw_output_path, fourcc, orig_fps, (width, height))

        if not writer.isOpened():
            cap.release()
            err = "Failed to initialize video writer. OpenCV video codec backend unavailable."
            logger.error(err)
            self._update_job_status("FAILED", error_message=err)
            raise RuntimeError(err)

        # Initialize CV components
        detector = ObjectDetector(
            model_name=config.YOLO_MODEL_NAME,
            device=config.DEVICE,
            conf_threshold=self.settings.confidence_threshold
        )
        tracker_mgr = TrackHistoryManager()
        
        signal_detector = SignalJumpDetector(
            stop_line=self.settings.calibration.stop_line,
            signal_roi=self.settings.calibration.signal_roi,
            signal_mode=self.settings.calibration.signal_mode,
            signal_rois=self.settings.calibration.signal_rois,
            stop_lines=self.settings.calibration.stop_lines
        )
        
        triple_detector = TripleRidingDetector(
            min_riders=config.MIN_RIDERS_FOR_ALERT,
            min_frames=config.TRIPLE_RIDING_MIN_FRAMES
        )
        
        helmet_detector = HelmetDetector(
            model_path=str(config.MODELS_DIR / config.HELMET_MODEL_NAME),
            min_confidence=config.HELMET_MIN_CONFIDENCE,
            smoothing_frames=config.HELMET_SMOOTHING_FRAMES
        )

        plate_engine = None
        if self.settings.enable_plate_ocr:
            plate_engine = PlateRecognitionEngine(ocr_min_conf=config.OCR_MIN_CONFIDENCE)

        self._update_job_status("PROCESSING", total_frames=total_frames)

        frame_idx = 0
        violations_count = 0
        start_time = time.time()

        try:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                frame_idx += 1
                if frame_idx % self.settings.frame_skip != 0:
                    continue

                # Ensure frame matches exact dimensions
                if frame.shape[1] != width or frame.shape[0] != height:
                    frame = cv2.resize(frame, (width, height))

                # Real object detection and ByteTrack tracking
                detections = detector.detect_and_track(frame, persist=True)
                tracker_mgr.update(detections, frame_idx)

                # Segregate detection classes
                vehicles = [d for d in detections if d.class_name in ["car", "motorcycle", "bus", "truck"]]
                motorcycles = [d for d in detections if d.class_name in ["motorcycle", "bicycle"]]
                persons = [d for d in detections if d.class_name == "person"]

                # 1. Signal Jump / Red-Light Violation Detection
                traffic_lights = [d for d in detections if d.class_name == "traffic light"]
                signal_states = signal_detector.determine_signal_states(frame, traffic_lights)
                if self.settings.enable_signal_jump:
                    trajectories = {t_id: tracker_mgr.get_trajectory(t_id) for t_id in tracker_mgr.tracks}
                    sig_violations = signal_detector.check_violations(
                        frame,
                        vehicles,
                        trajectories,
                        traffic_light_detections=traffic_lights,
                        signal_states=signal_states
                    )
                    for v in sig_violations:
                        t_id = v["tracking_id"]
                        v_created = self._handle_candidate_violation(
                            frame=frame,
                            event_type=EventType.RED_LIGHT_CROSSING,
                            track_id=t_id,
                            confidence=v["confidence"],
                            severity=Severity.HIGH,
                            box=v["box"],
                            notes=f"Vehicle #{t_id} crossed stop line {v['stop_line_index'] + 1} on RED light",
                            plate_engine=plate_engine
                        )
                        if v_created:
                            violations_count += 1
                            if t_id not in self.active_violations:
                                self.active_violations[t_id] = []
                            self.active_violations[t_id].append("SIGNAL JUMP")

                # 2. Triple Riding Detection
                moto_rider_counts = {}
                moto_riders_map = {}
                if self.settings.enable_triple_riding or self.settings.enable_helmet_check:
                    for moto in motorcycles:
                        m_id = moto.track_id
                        if m_id is not None:
                            riders = [p for p in persons if triple_detector.is_rider_on_motorcycle(p, moto)]
                            moto_riders_map[m_id] = riders

                if self.settings.enable_triple_riding:
                    tr_violations, moto_rider_counts = triple_detector.process_frame(motorcycles, persons)
                    for v in tr_violations:
                        t_id = v["tracking_id"]
                        v_created = self._handle_candidate_violation(
                            frame=frame,
                            event_type=EventType.TRIPLE_RIDING,
                            track_id=t_id,
                            confidence=v["confidence"],
                            severity=Severity.HIGH,
                            box=v["box"],
                            notes=v["notes"],
                            plate_engine=plate_engine
                        )
                        if v_created:
                            violations_count += 1
                            if t_id not in self.active_violations:
                                self.active_violations[t_id] = []
                            self.active_violations[t_id].append(f"TRIPLE RIDING ({v.get('rider_count', 3)} RIDERS)")

                # 3. Helmet Violation Detection
                if self.settings.enable_helmet_check:
                    helmet_violations = helmet_detector.process_riders(frame, moto_riders_map, motorcycles)
                    for v in helmet_violations:
                        t_id = v["tracking_id"]
                        v_created = self._handle_candidate_violation(
                            frame=frame,
                            event_type=EventType.NO_HELMET,
                            track_id=t_id,
                            confidence=v["confidence"],
                            severity=Severity.HIGH,
                            box=v["moto_box"],
                            notes=v["notes"],
                            plate_engine=plate_engine
                        )
                        if v_created:
                            violations_count += 1
                            if t_id not in self.active_violations:
                                self.active_violations[t_id] = []
                            self.active_violations[t_id].append("NO HELMET")

                # 4. Continuous Number Plate Detection & Recognition on Tracked Vehicles
                frame_plate_boxes = {}
                if plate_engine is not None and len(vehicles) > 0:
                    try:
                        detected_plates = plate_engine.detect_plates_yolo(frame)
                        for p_det in detected_plates:
                            p_box = p_det["box"]  # [px1, py1, px2, py2]
                            p_cx = (p_box[0] + p_box[2]) / 2.0
                            p_cy = (p_box[1] + p_box[3]) / 2.0

                            # Match plate with overlapping vehicle
                            matched_t_id = None
                            for v in vehicles:
                                vx1, vy1, vx2, vy2 = v.box
                                if vx1 <= p_cx <= vx2 and vy1 <= p_cy <= vy2:
                                    matched_t_id = v.track_id
                                    break

                            if matched_t_id is not None:
                                frame_plate_boxes[matched_t_id] = p_box
                                self.tracked_plate_boxes[matched_t_id] = p_box
                                # If vehicle plate text not yet recognized, run OCR
                                if matched_t_id not in self.recognized_plates or self.recognized_plates[matched_t_id] == "UNREADABLE":
                                    px1, py1, px2, py2 = [int(c) for c in p_box]
                                    hf, wf = frame.shape[:2]
                                    px1, py1 = max(0, px1), max(0, py1)
                                    px2, py2 = min(wf, px2), min(hf, py2)
                                    if px2 > px1 and py2 > py1:
                                        plate_crop = frame[py1:py2, px1:px2]
                                        norm_txt, raw_txt, ocr_conf = plate_engine.recognize_crop(plate_crop)
                                        if norm_txt != "UNREADABLE":
                                            self.recognized_plates[matched_t_id] = norm_txt
                    except Exception as e:
                        logger.debug("Error during continuous plate detection: %s", e)

                # Overlay traffic signal and stop line
                signal_detector.draw_overlay(frame, signal_states, traffic_lights)

                # Draw detections, track labels, plate bounding boxes, and active violation banners
                self._draw_detections(frame, detections, moto_rider_counts, tracker_mgr, frame_plate_boxes)

                # Draw top status HUD bar
                elapsed = time.time() - start_time
                current_fps = frame_idx / elapsed if elapsed > 0 else 0.0
                self._draw_hud(frame, frame_idx, total_frames, current_fps, len(vehicles), violations_count)

                # Privacy blur option if enabled
                if self.settings.privacy_blur:
                    self._apply_privacy_blur(frame, detections)

                # Write frame to raw video
                writer.write(frame)

                # Periodically update job progress
                progress = min(100.0, (frame_idx / total_frames) * 100.0)
                if frame_idx % 15 == 0 or frame_idx == total_frames:
                    self._update_job_status(
                        "PROCESSING",
                        progress=round(progress, 1),
                        current_frame=frame_idx,
                        total_frames=total_frames,
                        fps=round(current_fps, 1),
                        violations_count=violations_count
                    )
                    if self.progress_callback:
                        self.progress_callback(frame_idx, total_frames, current_fps, violations_count)

        except Exception as e:
            logger.exception("Error during video processing job %s: %s", self.job_id, e)
            self._update_job_status("FAILED", error_message=str(e))
            raise
        finally:
            cap.release()
            writer.release()

        # Step 2: Convert to web-streamable H.264 video with faststart
        logger.info("Muxing and encoding browser-compatible H.264 video for job %s...", self.job_id)
        self._encode_to_browser_h264(raw_output_path, self.output_path)

        # Step 3: Verify the generated output video
        self._verify_output_video(self.output_path)

        final_fps = round(frame_idx / (time.time() - start_time), 1) if (time.time() - start_time) > 0 else 0.0
        self._update_job_status(
            "COMPLETED",
            progress=100.0,
            current_frame=frame_idx,
            total_frames=frame_idx,
            fps=final_fps,
            violations_count=violations_count,
            annotated_video_path=self.output_path
        )
        logger.info("Job %s completed successfully. Annotated video verified at: %s", self.job_id, self.output_path)

        return {
            "job_id": self.job_id,
            "status": "COMPLETED",
            "total_frames": frame_idx,
            "violations_count": violations_count,
            "annotated_video_path": self.output_path
        }

    def _encode_to_browser_h264(self, raw_path: str, final_path: str):
        """Converts raw video into H.264 (yuv420p) with +faststart for HTML5 web streaming."""
        converted = False
        try:
            import imageio_ffmpeg
            ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
            cmd = [
                ffmpeg_exe, "-y",
                "-i", str(raw_path),
                "-c:v", "libx264",
                "-preset", "ultrafast",
                "-pix_fmt", "yuv420p",
                "-movflags", "+faststart",
                str(final_path)
            ]
            res = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
            if res.returncode == 0 and Path(final_path).exists() and Path(final_path).stat().st_size > 1000:
                converted = True
                Path(raw_path).unlink(missing_ok=True)
                logger.info("H.264 encoding completed successfully.")
            else:
                logger.warning("FFmpeg conversion failed (code %d): %s", res.returncode, res.stderr)
        except Exception as e:
            logger.warning("Error during H.264 FFmpeg conversion: %s", e)

        if not converted:
            logger.info("Falling back to direct raw video output.")
            shutil.move(raw_path, final_path)

    def _verify_output_video(self, video_path: str):
        """Verifies that the generated output video exists, is nonempty, and can be read by OpenCV."""
        p = Path(video_path)
        if not p.exists():
            raise RuntimeError(f"Output video file was not created: {video_path}")
        if p.stat().st_size < 1000:
            raise RuntimeError(f"Output video file is empty or corrupted ({p.stat().st_size} bytes): {video_path}")

        test_cap = cv2.VideoCapture(video_path)
        if not test_cap.isOpened():
            test_cap.release()
            raise RuntimeError(f"Output video cannot be opened by decoder: {video_path}")
        ret, _ = test_cap.read()
        test_cap.release()
        if not ret:
            logger.warning("Decoder could not read initial frame from output video: %s", video_path)

    def _handle_candidate_violation(
        self,
        frame: np.ndarray,
        event_type: EventType,
        track_id: Optional[int],
        confidence: float,
        severity: Severity,
        box: List[float],
        notes: str,
        plate_engine: Optional[PlateRecognitionEngine]
    ) -> bool:
        """Saves evidence images, runs plate OCR, and commits event to EventService."""
        # Deduplication check: avoid logging identical violation for same tracked vehicle
        if EventService.is_duplicate(event_type.value, track_id):
            return False

        event_id = str(uuid.uuid4())
        timestamp = datetime.now(timezone.utc).isoformat()
        cam_id = "DEMO_FEED" if self.is_demo else "CAM_UPLOAD_01"

        # 1. Save Evidence Frame
        evidence_filename = f"evidence_{event_type.value}_{event_id[:8]}.jpg"
        evidence_path = config.EVIDENCE_DIR / evidence_filename
        evidence_img = frame.copy()
        bx1, by1, bx2, by2 = [int(c) for c in box]
        cv2.rectangle(evidence_img, (bx1, by1), (bx2, by2), (0, 0, 255), 3)
        cv2.putText(
            evidence_img, f"VIOLATION: {event_type.value}",
            (bx1, max(25, by1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2
        )
        cv2.imwrite(str(evidence_path), evidence_img)

        # 2. Run License Plate Detection & OCR
        plate_text = "UNREADABLE"
        raw_plate = ""
        plate_conf = 0.0
        crop_path_str = None

        if plate_engine is not None:
            norm_txt, raw_txt, ocr_conf, plate_crop = plate_engine.recognize_plate(frame, box)
            plate_text = norm_txt
            raw_plate = raw_txt
            plate_conf = ocr_conf

            if plate_crop is not None and plate_crop.size > 0:
                crop_filename = f"plate_{event_id[:8]}.jpg"
                crop_path = config.EVIDENCE_DIR / crop_filename
                cv2.imwrite(str(crop_path), plate_crop)
                crop_path_str = str(crop_path)

        if track_id is not None and plate_text != "UNREADABLE":
            self.recognized_plates[track_id] = plate_text

        event_create = ViolationEventCreate(
            id=event_id,
            event_type=event_type,
            timestamp=timestamp,
            camera_id=cam_id,
            tracking_id=track_id,
            confidence=round(confidence, 3),
            severity=severity,
            status=ReviewStatus.PENDING_REVIEW,
            evidence_path=str(evidence_path),
            crop_path=crop_path_str,
            plate_text=plate_text,
            plate_raw_text=raw_plate,
            plate_confidence=round(plate_conf, 3),
            reviewer_notes=f"[{'DEMO' if self.is_demo else 'LIVE'}] {notes}",
            details={
                "box": box,
                "notes": notes,
                "job_id": self.job_id,
                "is_demo": self.is_demo
            }
        )

        EventService.create_event(event_create)
        return True

    def _draw_detections(
        self,
        frame: np.ndarray,
        detections: List[DetectionResult],
        moto_rider_counts: Dict[int, int],
        tracker_mgr: TrackHistoryManager,
        frame_plate_boxes: Optional[Dict[int, List[float]]] = None
    ):
        """Draws bounding boxes, labels, plate boxes, and trajectory paths with highlighted violations."""
        color_map = {
            "car": (255, 170, 0),
            "motorcycle": (0, 210, 255),
            "person": (210, 255, 0),
            "bus": (255, 80, 80),
            "truck": (210, 0, 210),
            "bicycle": (80, 255, 80)
        }

        for det in detections:
            t_id = det.track_id
            is_violator = t_id is not None and t_id in self.active_violations

            # Violator highlighted in bright RED
            color = (0, 0, 255) if is_violator else color_map.get(det.class_name, (200, 200, 200))
            box_thickness = 3 if is_violator else 2

            x1, y1, x2, y2 = [int(c) for c in det.box]
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, box_thickness)

            label = f"{det.class_name} #{t_id or '?'}"
            if det.class_name in ["motorcycle", "bicycle"] and t_id in moto_rider_counts:
                riders = moto_rider_counts[t_id]
                label += f" ({riders} riders)"

            cv2.putText(
                frame, label, (x1, max(18, y1 - 6)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2
            )

            # Display active violation banner above vehicle
            if is_violator:
                v_tags = " | ".join(self.active_violations[t_id])
                cv2.rectangle(frame, (x1, max(0, y1 - 30)), (min(frame.shape[1], x1 + 280), max(0, y1 - 6)), (0, 0, 220), -1)
                cv2.putText(
                    frame, f"! {v_tags}", (x1 + 4, max(12, y1 - 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 2
                )

            # Display recognized license plate box & badge
            if frame_plate_boxes and t_id is not None and t_id in frame_plate_boxes:
                p_box = frame_plate_boxes[t_id]
                px1, py1, px2, py2 = [int(c) for c in p_box]
                # High-visibility Emerald plate box
                cv2.rectangle(frame, (px1, py1), (px2, py2), (0, 255, 128), 2)
                plate = self.recognized_plates.get(t_id, "")
                badge_text = f"IND: {plate}" if (plate and plate != "UNREADABLE") else "PLATE"
                (tw, th), _ = cv2.getTextSize(badge_text, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
                by1 = max(0, py1 - th - 6)
                by2 = py1
                bx2 = min(frame.shape[1], px1 + tw + 8)
                cv2.rectangle(frame, (px1, by1), (bx2, by2), (15, 23, 42), -1)
                cv2.putText(
                    frame, badge_text, (px1 + 4, py1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 128), 1, cv2.LINE_AA
                )
            elif t_id is not None and t_id in self.recognized_plates:
                plate = self.recognized_plates[t_id]
                if plate != "UNREADABLE":
                    badge_text = f"IND: {plate}"
                    (tw, th), _ = cv2.getTextSize(badge_text, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
                    cv2.rectangle(frame, (x1, min(frame.shape[0], y2 + 2)), (min(frame.shape[1], x1 + tw + 10), min(frame.shape[0], y2 + th + 10)), (15, 23, 42), -1)
                    cv2.putText(
                        frame, badge_text, (x1 + 4, min(frame.shape[0] - 2, y2 + th + 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 128), 1, cv2.LINE_AA
                    )

            # Draw trajectory tail for tracked objects
            if t_id is not None:
                traj = tracker_mgr.get_trajectory(t_id)
                for i in range(1, len(traj)):
                    pt1 = (int(traj[i-1][0]), int(traj[i-1][1]))
                    pt2 = (int(traj[i][0]), int(traj[i][1]))
                    cv2.line(frame, pt1, pt2, color, 2)

    def _draw_hud(self, frame: np.ndarray, cur_frame: int, total_frames: int, fps: float, vehicles_count: int, viols_count: int):
        """Draws top HUD banner with frame, FPS, and violation counters."""
        h, w = frame.shape[:2]
        # Top banner background
        cv2.rectangle(frame, (0, 0), (w, 36), (15, 23, 42), -1)
        cv2.line(frame, (0, 36), (w, 36), (51, 65, 85), 1)

        hud_text = f"TRAFFICGUARD AI | Frame: {cur_frame}/{total_frames} | FPS: {fps:.1f} | Vehicles: {vehicles_count} | Violations: {viols_count}"
        if self.is_demo:
            hud_text += " [DEMO MODE]"

        cv2.putText(frame, hud_text, (16, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (241, 245, 249), 2)

    def _apply_privacy_blur(self, frame: np.ndarray, detections: List[DetectionResult]):
        """Blurs faces and plates in preview video when privacy mode is enabled."""
        h, w = frame.shape[:2]
        for det in detections:
            if det.class_name == "person":
                y1 = max(0, int(det.y1))
                y2 = min(h, int(det.y1 + 0.25 * det.height))
                x1 = max(0, int(det.x1))
                x2 = min(w, int(det.x2))
                if y2 > y1 and x2 > x1:
                    roi = frame[y1:y2, x1:x2]
                    frame[y1:y2, x1:x2] = cv2.GaussianBlur(roi, (23, 23), 30)
            elif det.class_name in ["car", "motorcycle", "bus", "truck"]:
                y1 = max(0, int(det.y1 + 0.65 * det.height))
                y2 = min(h, int(det.y2))
                x1 = max(0, int(det.x1 + 0.2 * det.width))
                x2 = min(w, int(det.x2 - 0.2 * det.width))
                if y2 > y1 and x2 > x1:
                    roi = frame[y1:y2, x1:x2]
                    frame[y1:y2, x1:x2] = cv2.GaussianBlur(roi, (25, 25), 30)

    def _update_job_status(
        self,
        status: str,
        progress: float = 0.0,
        current_frame: int = 0,
        total_frames: int = 0,
        fps: float = 0.0,
        violations_count: int = 0,
        annotated_video_path: Optional[str] = None,
        error_message: Optional[str] = None
    ):
        conn = get_db_connection()
        try:
            now = datetime.now(timezone.utc).isoformat()
            with conn:
                conn.execute("""
                INSERT INTO processing_jobs (
                    id, source_filename, status, progress, current_frame, total_frames,
                    fps, violations_count, annotated_video_path, error_message, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    status = excluded.status,
                    progress = excluded.progress,
                    current_frame = excluded.current_frame,
                    total_frames = excluded.total_frames,
                    fps = excluded.fps,
                    violations_count = excluded.violations_count,
                    annotated_video_path = COALESCE(excluded.annotated_video_path, processing_jobs.annotated_video_path),
                    error_message = excluded.error_message,
                    updated_at = excluded.updated_at
                """, (
                    self.job_id, self.video_source, status, progress, current_frame, total_frames,
                    fps, violations_count, annotated_video_path, error_message, now, now
                ))
        except Exception as e:
            logger.error("Failed to update job status in DB: %s", e)
        finally:
            conn.close()
