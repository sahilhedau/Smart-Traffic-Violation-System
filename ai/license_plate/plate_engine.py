import re
import cv2
import numpy as np
import logging
from typing import Tuple, Optional, Dict, Any, List
from pathlib import Path

logger = logging.getLogger("traffic_system.plate_ocr")

class PlateRecognitionEngine:
    def __init__(self, ocr_min_conf: float = 0.25):
        self.ocr_min_conf = ocr_min_conf
        self.reader = None
        self._init_ocr()

    def _init_ocr(self):
        try:
            import easyocr
            logger.info("Initializing EasyOCR (English/Alphanumeric)...")
            # Uses GPU if available, else CPU
            self.reader = easyocr.Reader(['en'], gpu=True)
            logger.info("EasyOCR initialized successfully with GPU.")
        except Exception as e:
            logger.warning("GPU EasyOCR initialization failed (%s). Trying CPU...", e)
            try:
                import easyocr
                self.reader = easyocr.Reader(['en'], gpu=False)
                logger.info("EasyOCR initialized on CPU.")
            except Exception as ex:
                logger.error("EasyOCR could not be initialized: %s", ex)
                self.reader = None

    def preprocess_plate_image(self, plate_crop: np.ndarray) -> np.ndarray:
        """
        Preprocesses plate crop for improved OCR character recognition:
        - Resize to standard height
        - Grayscale conversion
        - Bilateral filter to smooth noise while preserving edges
        - CLAHE contrast enhancement
        - Adaptive / Otsu thresholding
        """
        if plate_crop is None or plate_crop.size == 0:
            return plate_crop

        h, w = plate_crop.shape[:2]
        # Target height ~64-80 pixels for clear OCR
        target_h = 70
        aspect = w / float(h)
        target_w = int(target_h * aspect)
        resized = cv2.resize(plate_crop, (target_w, target_h), interpolation=cv2.INTER_CUBIC)

        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        denoised = cv2.bilateralFilter(gray, 9, 75, 75)

        # Contrast Limited Adaptive Histogram Equalization
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(denoised)

        return enhanced

    def find_plate_candidate_region(self, vehicle_crop: np.ndarray) -> Optional[np.ndarray]:
        """
        Extracts the lower portion of the vehicle where license plates are mounted,
        and locates high-contrast rectangular plate candidates.
        """
        if vehicle_crop is None or vehicle_crop.size == 0:
            return None

        vh, vw = vehicle_crop.shape[:2]
        # Plate is almost always in the lower 40% of the vehicle
        lower_y1 = int(vh * 0.55)
        lower_crop = vehicle_crop[lower_y1:vh, :]

        if lower_crop.size == 0:
            return vehicle_crop

        gray = cv2.cvtColor(lower_crop, cv2.COLOR_BGR2GRAY)
        # Morphological gradient to detect plate text edges
        morph_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (13, 5))
        tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, morph_kernel)

        contours, _ = cv2.findContours(tophat, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        best_box = None
        best_score = 0.0

        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            aspect_ratio = float(w) / float(h) if h > 0 else 0
            area = w * h
            # License plates typically have aspect ratio between 2.0 and 6.0
            if 1.8 <= aspect_ratio <= 5.8 and area > 600:
                score = area
                if score > best_score:
                    best_score = score
                    best_box = (x, y, w, h)

        if best_box is not None:
            x, y, w, h = best_box
            # Pad slightly
            px = max(0, int(w * 0.08))
            py = max(0, int(h * 0.08))
            cx1 = max(0, x - px)
            cy1 = max(0, y - py)
            cx2 = min(lower_crop.shape[1], x + w + px)
            cy2 = min(lower_crop.shape[0], y + h + py)
            return lower_crop[cy1:cy2, cx1:cx2]

        return lower_crop

    def normalize_plate_text(self, raw_text: str) -> str:
        """
        Cleans and normalizes plate text:
        Removes special characters, converts to uppercase.
        Checks for valid Indian vehicle plate format patterns.
        """
        if not raw_text:
            return "UNREADABLE"

        # Remove spaces, hyphens, punctuation
        cleaned = re.sub(r'[^A-Za-z0-9]', '', raw_text).upper()

        if len(cleaned) < 4:
            return "UNREADABLE"

        # Common Indian format: State code (2 letters) + RTO (1-2 digits) + Series (0-3 letters) + Number (4 digits)
        # e.g., MH12AB1234 or DL1CAA1111 or HR26DQ5551
        indian_pattern = r'^[A-Z]{2}[0-9]{1,2}[A-Z]{0,3}[0-9]{3,4}$'
        if re.match(indian_pattern, cleaned):
            return cleaned

        # General alphanumeric plate pattern
        if 4 <= len(cleaned) <= 12 and any(c.isdigit() for c in cleaned) and any(c.isalpha() for c in cleaned):
            return cleaned

        return "UNREADABLE"

    def recognize_plate(
        self,
        frame: np.ndarray,
        vehicle_box: List[float]
    ) -> Tuple[str, str, float, Optional[np.ndarray]]:
        """
        Runs the full plate recognition pipeline for a given vehicle:
        Returns: (normalized_text, raw_text, confidence, plate_crop)
        """
        h, w = frame.shape[:2]
        vx1 = max(0, int(vehicle_box[0]))
        vy1 = max(0, int(vehicle_box[1]))
        vx2 = min(w, int(vehicle_box[2]))
        vy2 = min(h, int(vehicle_box[3]))

        if vx2 <= vx1 or vy2 <= vy1:
            return "UNREADABLE", "", 0.0, None

        vehicle_crop = frame[vy1:vy2, vx1:vx2]
        plate_candidate = self.find_plate_candidate_region(vehicle_crop)
        if plate_candidate is None:
            plate_candidate = vehicle_crop

        preprocessed = self.preprocess_plate_image(plate_candidate)

        raw_text = ""
        confidence = 0.0

        if self.reader is not None:
            try:
                # Run OCR on preprocessed image
                ocr_results = self.reader.readtext(
                    preprocessed,
                    detail=1,
                    paragraph=False,
                    allowlist='ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
                )

                if ocr_results:
                    # Pick highest confidence or join sequential texts
                    texts = []
                    confs = []
                    for bbox, text, conf in ocr_results:
                        if conf >= self.ocr_min_conf and text.strip():
                            texts.append(text.strip())
                            confs.append(conf)

                    if texts:
                        raw_text = " ".join(texts)
                        confidence = float(np.mean(confs))
            except Exception as e:
                logger.error("Error during plate OCR recognition: %s", e)

        normalized = self.normalize_plate_text(raw_text)
        return normalized, raw_text, round(confidence, 3), plate_candidate
