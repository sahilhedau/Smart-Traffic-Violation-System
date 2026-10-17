import os
import cv2
import time
import uuid
import logging
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime, timezone

from fastapi import APIRouter, UploadFile, File, HTTPException, Form
from fastapi.responses import FileResponse

from backend.app.config import config
from backend.app.database.db import get_db_connection
from backend.app.schemas.image import (
    ImageDetectionResponse,
    ImageDetectionBox,
    PlateDetectionResult,
    HelmetDetectionItem,
    ImageViolationItem
)
from ai.detection.detector import ObjectDetector, DetectionResult
from ai.helmet.helmet_engine import HelmetDetector
from ai.license_plate.plate_engine import PlateRecognitionEngine
from ai.triple_riding.triple_riding_engine import TripleRidingDetector

logger = logging.getLogger("traffic_system.api.image")
router = APIRouter(prefix="/image", tags=["image"])

ALLOWED_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
MAX_IMAGE_SIZE = 30 * 1024 * 1024  # 30 MB

# Cached detectors
_detector: Optional[ObjectDetector] = None
_helmet_detector: Optional[HelmetDetector] = None
_plate_engine: Optional[PlateRecognitionEngine] = None
_triple_engine: Optional[TripleRidingDetector] = None

def get_detectors():
    global _detector, _helmet_detector, _plate_engine, _triple_engine
    if _detector is None:
        _detector = ObjectDetector(
            model_name=str(config.MODELS_DIR / config.YOLO_MODEL_NAME),
            device=config.DEVICE,
            conf_threshold=config.CONFIDENCE_THRESHOLD
        )
    if _helmet_detector is None:
        _helmet_detector = HelmetDetector(
            model_path=str(config.MODELS_DIR / config.HELMET_MODEL_NAME),
            min_confidence=config.HELMET_MIN_CONFIDENCE
        )
    if _plate_engine is None:
        _plate_engine = PlateRecognitionEngine(
            model_path=str(config.MODELS_DIR / config.PLATE_MODEL_NAME),
            ocr_min_conf=config.OCR_MIN_CONFIDENCE
        )
    if _triple_engine is None:
        _triple_engine = TripleRidingDetector(
            min_riders=config.MIN_RIDERS_FOR_ALERT,
            min_frames=1
        )
    return _detector, _helmet_detector, _plate_engine, _triple_engine

@router.post("/detect", response_model=ImageDetectionResponse)
async def detect_image(
    file: UploadFile = File(...),
    save_violations_to_db: bool = Form(True)
):
    """
    Upload an image for instant YOLO detection, helmet verification, and license plate OCR.
    Returns detected objects, bounding boxes, plate text, and any traffic violations.
    """
    ext = Path(file.filename).suffix.lower()
    if ext not in ALLOWED_IMAGE_EXTS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported image format '{ext}'. Allowed: {', '.join(ALLOWED_IMAGE_EXTS)}"
        )

    content = await file.read()
    if len(content) > MAX_IMAGE_SIZE:
        raise HTTPException(status_code=413, detail="Image file exceeds maximum limit of 30 MB")

    nparr = np.frombuffer(content, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if frame is None:
        raise HTTPException(status_code=400, detail="Corrupted or invalid image file")

    start_time = time.time()
    h, w = frame.shape[:2]
    annotated = frame.copy()

    detector, helmet_detector, plate_engine, triple_engine = get_detectors()

    # 1. Base YOLO object detection
    detections: List[DetectionResult] = detector.detect_and_track(frame, persist=False)

    vehicles = [d for d in detections if d.class_name in ["car", "motorcycle", "bus", "truck"]]
    motorcycles = [d for d in detections if d.class_name == "motorcycle"]
    persons = [d for d in detections if d.class_name == "person"]

    out_detections: List[ImageDetectionBox] = []
    out_helmets: List[HelmetDetectionItem] = []
    out_plates: List[PlateDetectionResult] = []
    out_violations: List[ImageViolationItem] = []

    # Draw vehicles & persons
    for d in detections:
        out_detections.append(ImageDetectionBox(
            box=d.box,
            class_name=d.class_name,
            confidence=d.confidence
        ))
        bx1, by1, bx2, by2 = map(int, d.box)
        color = (255, 180, 0) if d.class_name == "motorcycle" else (0, 200, 100)
        cv2.rectangle(annotated, (bx1, by1), (bx2, by2), color, 2)
        cv2.putText(
            annotated,
            f"{d.class_name} {d.confidence:.2f}",
            (bx1, max(15, by1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            2
        )

    # 2. Triple Riding & Helmet Analysis on Motorcycles
    moto_riders_map = triple_engine.associate_riders(motorcycles, persons)

    for m_idx, moto in enumerate(motorcycles):
        riders = moto_riders_map.get(m_idx, [])
        mx1, my1, mx2, my2 = map(int, moto.box)

        # Check Triple Riding (>= 3 riders)
        if len(riders) >= config.MIN_RIDERS_FOR_ALERT:
            viol = ImageViolationItem(
                event_type="TRIPLE_RIDING",
                confidence=moto.confidence,
                severity="HIGH",
                description=f"Triple riding detected ({len(riders)} riders on motorcycle)",
                associated_box=moto.box
            )
            out_violations.append(viol)
            cv2.rectangle(annotated, (mx1, my1), (mx2, my2), (0, 0, 255), 3)
            cv2.putText(
                annotated,
                f"ALERT: TRIPLE RIDING ({len(riders)})",
                (mx1, min(h - 10, my2 + 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 255),
                2
            )

        # Check Helmets on each associated rider
        for r in riders:
            status, conf = helmet_detector.classify_rider(frame, r.box)
            rx1, ry1, rx2, ry2 = map(int, r.box)
            head_y2 = min(h, int(ry1 + 0.30 * (ry2 - ry1)))
            head_box = [float(rx1), float(ry1), float(rx2), float(head_y2)]

            is_viol = (status == "NO_HELMET" and conf >= config.HELMET_MIN_CONFIDENCE)
            out_helmets.append(HelmetDetectionItem(
                box=head_box,
                status=status,
                confidence=round(conf, 3),
                is_violation=is_viol
            ))

            if is_viol:
                out_violations.append(ImageViolationItem(
                    event_type="NO_HELMET",
                    confidence=round(conf, 3),
                    severity="HIGH",
                    description=f"Rider without helmet detected (conf: {conf:.2f})",
                    associated_box=r.box
                ))
                # Red box for no helmet
                cv2.rectangle(annotated, (rx1, ry1), (rx2, head_y2), (0, 0, 255), 2)
                cv2.putText(
                    annotated,
                    f"NO HELMET {conf:.2f}",
                    (rx1, max(15, ry1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 0, 255),
                    2
                )
            elif status == "HELMET":
                cv2.rectangle(annotated, (rx1, ry1), (rx2, head_y2), (0, 255, 0), 2)
                cv2.putText(
                    annotated,
                    f"HELMET {conf:.2f}",
                    (rx1, max(15, ry1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    2
                )

    # 3. Direct Helmet Frame Detection (if specialized YOLO model is loaded)
    frame_helmets = helmet_detector.detect_in_frame(frame)
    for fh in frame_helmets:
        box = fh["box"]
        conf = fh["confidence"]
        is_viol = fh["is_violation"]
        status = "NO_HELMET" if is_viol else "HELMET"
        out_helmets.append(HelmetDetectionItem(
            box=box,
            status=status,
            confidence=conf,
            is_violation=is_viol
        ))
        if is_viol:
            out_violations.append(ImageViolationItem(
                event_type="NO_HELMET",
                confidence=conf,
                severity="HIGH",
                description=f"Rider without helmet detected (conf: {conf:.2f})",
                associated_box=box
            ))
        hx1, hy1, hx2, hy2 = map(int, box)
        hcolor = (0, 0, 255) if is_viol else (0, 255, 0)
        cv2.rectangle(annotated, (hx1, hy1), (hx2, hy2), hcolor, 2)
        cv2.putText(
            annotated,
            f"{fh['class_name']} {conf:.2f}",
            (hx1, max(15, hy1 - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            hcolor,
            2
        )

    # 4. License Plate Detection & OCR for Vehicles
    for v in vehicles:
        norm_plate, raw_plate, ocr_conf, crop = plate_engine.recognize_plate(frame, v.box)
        if crop is not None:
            is_verified = plate_engine.is_verified_plate(norm_plate, ocr_conf)
            out_plates.append(PlateDetectionResult(
                vehicle_box=v.box,
                plate_text=norm_plate,
                raw_text=raw_plate,
                confidence=ocr_conf,
                is_verified=is_verified
            ))
            # Draw exact plate bounding box on vehicle if detected
            if getattr(plate_engine, "last_plate_box", None) is not None:
                px1, py1, px2, py2 = map(int, plate_engine.last_plate_box)
                cv2.rectangle(annotated, (px1, py1), (px2, py2), (0, 255, 128), 2)

            # Draw plate tag near bottom of vehicle
            vx1, vy1, vx2, vy2 = map(int, v.box)
            plate_label = f"IND: {norm_plate}" if is_verified else f"PLATE: {norm_plate}"
            cv2.putText(
                annotated,
                plate_label,
                (vx1, min(h - 5, vy2 - 5)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 255, 128),
                2
            )

    inference_ms = (time.time() - start_time) * 1000.0

    # 5. Save Annotated Image
    img_id = str(uuid.uuid4())[:8]
    annotated_filename = f"annotated_img_{img_id}_{Path(file.filename).name}"
    annotated_path = config.EVIDENCE_DIR / annotated_filename
    cv2.imwrite(str(annotated_path), annotated)

    # 6. Database recording for violations
    if save_violations_to_db and out_violations:
        conn = get_db_connection()
        try:
            now = datetime.now(timezone.utc).isoformat()
            with conn:
                for viol in out_violations:
                    event_id = str(uuid.uuid4())
                    conn.execute("""
                    INSERT INTO violation_events (
                        id, event_type, timestamp, camera_id, tracking_id, confidence,
                        severity, status, evidence_path, plate_text, created_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        event_id, viol.event_type, now, "SNAPSHOT_UPLOAD", None,
                        viol.confidence, viol.severity, "PENDING_REVIEW",
                        str(annotated_path),
                        out_plates[0].plate_text if out_plates else None,
                        now, now
                    ))
        finally:
            conn.close()

    counts = {
        "vehicles": len(vehicles),
        "persons": len(persons),
        "motorcycles": len(motorcycles),
        "helmets_detected": sum(1 for h in out_helmets if h.status == "HELMET"),
        "no_helmets_detected": sum(1 for h in out_helmets if h.status == "NO_HELMET"),
        "plates_detected": len(out_plates),
        "violations": len(out_violations)
    }

    return ImageDetectionResponse(
        status="success",
        filename=file.filename,
        width=w,
        height=h,
        inference_time_ms=round(inference_ms, 1),
        annotated_image_url=f"/api/image/evidence/{annotated_filename}",
        counts=counts,
        detections=out_detections,
        helmet_detections=out_helmets,
        plate_detections=out_plates,
        violations=out_violations
    )

@router.get("/evidence/{filename}")
def get_annotated_evidence(filename: str):
    """Serves the annotated evidence image file."""
    path = config.EVIDENCE_DIR / filename
    if not path.exists():
        raise HTTPException(status_code=404, detail="Evidence image file not found")
    return FileResponse(path, media_type="image/jpeg")
