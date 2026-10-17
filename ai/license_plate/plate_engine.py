import re
import cv2
import numpy as np
import logging
from typing import Tuple, Optional, Dict, Any, List
from pathlib import Path
from backend.app.config import config

logger = logging.getLogger("traffic_system.plate_ocr")

class PlateRecognitionEngine:
    def __init__(
        self,
        model_path: Optional[str] = None,
        ocr_min_conf: float = 0.25,
        plate_min_conf: float = 0.35
    ):
        self.ocr_min_conf = ocr_min_conf
        self.plate_min_conf = plate_min_conf
        self.reader = None
        self.plate_model = None
        
        # 1. Initialize YOLO License Plate Detector if available
        if not model_path:
            model_path = str(config.MODELS_DIR / config.PLATE_MODEL_NAME)
        
        if model_path and Path(model_path).exists():
            try:
                from ultralytics import YOLO
                logger.info("Loading specialized YOLO plate detection model from %s...", model_path)
                self.plate_model = YOLO(model_path)
                logger.info("YOLO plate detection model loaded successfully.")
            except Exception as e:
                logger.warning("Could not load YOLO plate model (%s): %s. Using heuristic region detector.", model_path, e)
        else:
            logger.info("No YOLO plate model found at '%s'. Using heuristic region candidate detector.", model_path)

        # 2. Initialize EasyOCR
        self._init_ocr()

    def _init_ocr(self):
        try:
            import easyocr
            logger.info("Initializing EasyOCR (English/Alphanumeric)...")
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
        """
        if plate_crop is None or plate_crop.size == 0:
            return plate_crop

        h, w = plate_crop.shape[:2]
        target_h = 70
        aspect = w / float(h) if h > 0 else 1.0
        target_w = max(10, int(target_h * aspect))
        resized = cv2.resize(plate_crop, (target_w, target_h), interpolation=cv2.INTER_CUBIC)

        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        denoised = cv2.bilateralFilter(gray, 9, 75, 75)

        # Contrast Limited Adaptive Histogram Equalization
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(denoised)

        return enhanced

    def _preprocess_plate_variants(self, plate_crop: np.ndarray) -> List[np.ndarray]:
        """Prepare complementary OCR inputs for small, low-contrast plate crops."""
        enhanced = self.preprocess_plate_image(plate_crop)
        if enhanced is None or enhanced.size == 0:
            return []

        enlarged = cv2.resize(
            enhanced,
            None,
            fx=2.0,
            fy=2.0,
            interpolation=cv2.INTER_CUBIC
        )
        thresholded = cv2.adaptiveThreshold(
            enlarged,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            31,
            11
        )
        return [enhanced, enlarged, thresholded]

    def detect_plates_yolo(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """
        Runs YOLO inference specifically to detect license plate bounding boxes.
        Returns list of detections: [{'box': [x1, y1, x2, y2], 'confidence': float}]
        """
        plates = []
        if self.plate_model is None or image is None or image.size == 0:
            return plates

        try:
            results = self.plate_model.predict(
                source=image,
                conf=self.plate_min_conf,
                verbose=False
            )
            if not results or len(results) == 0 or results[0].boxes is None:
                return plates

            boxes = results[0].boxes.xyxy.cpu().numpy()
            confs = results[0].boxes.conf.cpu().numpy()

            for box, conf in zip(boxes, confs):
                plates.append({
                    "box": [round(float(c), 1) for c in box],
                    "confidence": round(float(conf), 3)
                })
        except Exception as e:
            logger.error("Error during YOLO plate inference: %s", e)

        return plates

    def find_plate_candidate_region(self, vehicle_crop: np.ndarray) -> Optional[np.ndarray]:
        """
        Locates plate using YOLO if model is loaded; otherwise uses morphological gradient candidate extraction.
        """
        if vehicle_crop is None or vehicle_crop.size == 0:
            return None

        # 1. Try YOLO plate detector first
        if self.plate_model is not None:
            detected = self.detect_plates_yolo(vehicle_crop)
            if detected:
                # Pick the highest confidence detection
                best = max(detected, key=lambda d: d["confidence"])
                x1, y1, x2, y2 = map(int, best["box"])
                vh, vw = vehicle_crop.shape[:2]
                pad_x = max(1, int((x2 - x1) * 0.04))
                pad_y = max(1, int((y2 - y1) * 0.10))
                x1, y1 = max(0, x1 - pad_x), max(0, y1 - pad_y)
                x2, y2 = min(vw, x2 + pad_x), min(vh, y2 + pad_y)
                if x2 > x1 and y2 > y1:
                    return vehicle_crop[y1:y2, x1:x2]

        # 2. Heuristic fallback: lower 45% of vehicle
        vh, vw = vehicle_crop.shape[:2]
        lower_y1 = int(vh * 0.55)
        lower_crop = vehicle_crop[lower_y1:vh, :]

        if lower_crop.size == 0:
            return vehicle_crop

        gray = cv2.cvtColor(lower_crop, cv2.COLOR_BGR2GRAY)
        morph_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (13, 5))
        tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, morph_kernel)

        contours, _ = cv2.findContours(tophat, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        best_box = None
        best_score = 0.0

        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            aspect_ratio = float(w) / float(h) if h > 0 else 0
            area = w * h
            if 1.8 <= aspect_ratio <= 5.8 and area > 600:
                if area > best_score:
                    best_score = area
                    best_box = (x, y, w, h)

        if best_box is not None:
            x, y, w, h = best_box
            px = max(0, int(w * 0.08))
            py = max(0, int(h * 0.08))
            cx1 = max(0, x - px)
            cy1 = max(0, y - py)
            cx2 = min(lower_crop.shape[1], x + w + px)
            cy2 = min(lower_crop.shape[0], y + h + py)
            return lower_crop[cy1:cy2, cx1:cx2]

        return lower_crop

    INDIAN_STATES = {
        "AN", "AP", "AR", "AS", "BR", "CG", "CH", "DD", "DL", "DN", "GA", "GJ",
        "HP", "HR", "JH", "JK", "KA", "KL", "LA", "LD", "MH", "ML", "MN", "MP",
        "MZ", "NL", "OD", "PB", "PY", "RJ", "SK", "TN", "TR", "TS", "UK", "UP", "WB"
    }

    # Common OCR character confusions between letters and digits
    DIGIT_TO_LETTER = {'0': 'O', '1': 'I', '2': 'Z', '4': 'A', '5': 'S', '6': 'G', '8': 'B'}
    LETTER_TO_DIGIT = {'O': '0', 'D': '0', 'I': '1', 'L': '1', 'Z': '2', 'A': '4', 'S': '5', 'G': '6', 'B': '8'}

    def correct_indian_plate(self, cleaned: str) -> str:
        """
        Applies positional character correction for Indian vehicle license plates:
        Format: [State: 2 chars] [RTO: 1-2 digits] [Series: 0-3 chars] [Number: 3-4 digits]
        """
        if len(cleaned) < 6 or len(cleaned) > 11:
            return cleaned

        chars = list(cleaned)
        # 1. First two characters must be State Letters
        for i in range(min(2, len(chars))):
            if chars[i] in self.DIGIT_TO_LETTER:
                chars[i] = self.DIGIT_TO_LETTER[chars[i]]

        # Check if first two chars match or near-match a known Indian state
        cand_state = "".join(chars[:2])
        if cand_state not in self.INDIAN_STATES:
            # Check single-character edit
            for s in self.INDIAN_STATES:
                if (chars[0] == s[0] and chars[1] in self.DIGIT_TO_LETTER.values()) or \
                   (chars[1] == s[1] and chars[0] in self.DIGIT_TO_LETTER.values()):
                    chars[0], chars[1] = s[0], s[1]
                    break

        # 2. Next 1 or 2 characters must be Digits (RTO)
        if len(chars) >= 4:
            if chars[2] in self.LETTER_TO_DIGIT:
                chars[2] = self.LETTER_TO_DIGIT[chars[2]]
            if len(chars) >= 5 and chars[3] in self.LETTER_TO_DIGIT and (chars[4].isalpha() or chars[4] in self.DIGIT_TO_LETTER):
                chars[3] = self.LETTER_TO_DIGIT[chars[3]]

        # 3. Last 3 or 4 characters must be Digits (Vehicle Registration Number)
        tail_len = 4 if len(chars) >= 9 else 3
        for idx in range(len(chars) - tail_len, len(chars)):
            if idx >= 0 and chars[idx] in self.LETTER_TO_DIGIT:
                chars[idx] = self.LETTER_TO_DIGIT[chars[idx]]

        return "".join(chars)

    def normalize_plate_text(self, raw_text: str) -> str:
        """
        Cleans and normalizes plate text:
        Removes special characters, converts to uppercase.
        Applies Indian ALPR format-aware character correction.
        """
        if not raw_text:
            return "UNREADABLE"

        cleaned = re.sub(r'[^A-Za-z0-9]', '', raw_text).upper()

        if len(cleaned) < 4:
            return "UNREADABLE"

        # Apply Indian ALPR positional correction
        corrected = self.correct_indian_plate(cleaned)

        # Standard Indian format: State code (2 letters) + RTO (1-2 digits) + Series (0-3 letters) + Number (3-4 digits)
        indian_pattern = r'^[A-Z]{2}[0-9]{1,2}[A-Z]{0,3}[0-9]{3,4}$'
        if re.match(indian_pattern, corrected):
            return corrected

        if re.match(indian_pattern, cleaned):
            return cleaned

        # General alphanumeric plate pattern (fallback)
        if 4 <= len(cleaned) <= 12 and any(c.isdigit() for c in cleaned) and any(c.isalpha() for c in cleaned):
            return cleaned

        return "UNREADABLE"

    def is_verified_plate(self, normalized_text: str, confidence: float) -> bool:
        """Returns True if plate text matches verified format with sufficient OCR confidence."""
        if not normalized_text or normalized_text == "UNREADABLE":
            return False
        return confidence >= self.ocr_min_conf

    def recognize_crop(self, plate_crop: np.ndarray) -> Tuple[str, str, float]:
        """
        Runs OCR directly on a cropped license plate image.
        Returns: (normalized_text, raw_text, confidence)
        """
        if plate_crop is None or plate_crop.size == 0 or self.reader is None:
            return "UNREADABLE", "", 0.0

        candidates = []
        for variant in self._preprocess_plate_variants(plate_crop):
            try:
                ocr_results = self.reader.readtext(
                    variant,
                    detail=1,
                    paragraph=False,
                    allowlist='ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
                )
            except Exception as error:
                logger.debug("Error during plate crop OCR pass: %s", error)
                continue

            recognized = [
                (text.strip(), float(confidence))
                for _, text, confidence in ocr_results or []
                if text.strip() and confidence >= self.ocr_min_conf
            ]
            if not recognized:
                continue
            raw_text = " ".join(text for text, _ in recognized)
            confidence = float(np.mean([score for _, score in recognized]))
            normalized = self.normalize_plate_text(raw_text)
            if normalized == "UNREADABLE":
                continue
            is_indian_format = bool(re.fullmatch(
                r"[A-Z]{2}[0-9]{1,2}[A-Z]{0,3}[0-9]{3,4}",
                normalized
            ))
            candidates.append((normalized, raw_text, confidence, is_indian_format))

        if not candidates:
            return "UNREADABLE", "", 0.0

        counts = {}
        for normalized, _, _, _ in candidates:
            counts[normalized] = counts.get(normalized, 0) + 1
        best = max(
            candidates,
            key=lambda candidate: (
                candidate[3],
                counts[candidate[0]],
                candidate[2]
            )
        )
        return best[0], best[1], round(best[2], 3)

    def find_plate_with_box(self, vehicle_crop: np.ndarray) -> Tuple[Optional[np.ndarray], Optional[List[float]]]:
        """
        Locates license plate in vehicle crop and returns: (plate_crop, [x1, y1, x2, y2] relative to vehicle_crop)
        """
        if vehicle_crop is None or vehicle_crop.size == 0:
            return None, None

        # 1. Try YOLO plate detector
        if self.plate_model is not None:
            detected = self.detect_plates_yolo(vehicle_crop)
            if detected:
                best = max(detected, key=lambda d: d["confidence"])
                x1, y1, x2, y2 = map(int, best["box"])
                vh, vw = vehicle_crop.shape[:2]
                pad_x = max(1, int((x2 - x1) * 0.04))
                pad_y = max(1, int((y2 - y1) * 0.10))
                x1, y1 = max(0, x1 - pad_x), max(0, y1 - pad_y)
                x2, y2 = min(vw, x2 + pad_x), min(vh, y2 + pad_y)
                if x2 > x1 and y2 > y1:
                    return vehicle_crop[y1:y2, x1:x2], [float(x1), float(y1), float(x2), float(y2)]

        # 2. Heuristic fallback
        candidate = self.find_plate_candidate_region(vehicle_crop)
        return candidate, None

    def recognize_plate(
        self,
        frame: np.ndarray,
        vehicle_box: List[float]
    ) -> Tuple[str, str, float, Optional[np.ndarray]]:
        """
        Runs the full plate recognition pipeline for a given vehicle:
        Returns: (normalized_text, raw_text, confidence, plate_crop)
        Also updates self.last_plate_box with global frame coordinates.
        """
        self.last_plate_box = None
        h, w = frame.shape[:2]
        vx1 = max(0, int(vehicle_box[0]))
        vy1 = max(0, int(vehicle_box[1]))
        vx2 = min(w, int(vehicle_box[2]))
        vy2 = min(h, int(vehicle_box[3]))

        if vx2 <= vx1 or vy2 <= vy1:
            return "UNREADABLE", "", 0.0, None

        vehicle_crop = frame[vy1:vy2, vx1:vx2]
        plate_candidate, local_box = self.find_plate_with_box(vehicle_crop)
        if plate_candidate is None:
            plate_candidate = vehicle_crop

        if local_box is not None:
            self.last_plate_box = [
                vx1 + local_box[0],
                vy1 + local_box[1],
                vx1 + local_box[2],
                vy1 + local_box[3]
            ]

        normalized, raw_text, confidence = self.recognize_crop(plate_candidate)
        return normalized, raw_text, confidence, plate_candidate

    def detect_and_recognize_all(self, frame: np.ndarray, vehicle_boxes: List[List[float]]) -> List[Dict[str, Any]]:
        """
        Recognizes plates across multiple vehicles in an image/frame.
        Returns structured list of detected plates with coordinates, OCR text, and verification state.
        """
        results = []
        for v_box in vehicle_boxes:
            norm_text, raw_text, conf, crop = self.recognize_plate(frame, v_box)
            if crop is not None:
                results.append({
                    "vehicle_box": [round(c, 1) for c in v_box],
                    "plate_text": norm_text,
                    "raw_text": raw_text,
                    "confidence": conf,
                    "is_verified": self.is_verified_plate(norm_text, conf),
                    "crop": crop
                })
        return results
