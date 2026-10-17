import logging
from typing import Dict, Any
from backend.app.config import config
from backend.app.database.db import get_setting, save_setting
from backend.app.schemas.settings import SystemSettings, CalibrationSettings

logger = logging.getLogger("traffic_system.settings_service")

class SettingsService:
    @staticmethod
    def get_settings() -> SystemSettings:
        saved = get_setting("system_settings", None)
        if saved:
            try:
                return SystemSettings(**saved)
            except Exception as e:
                logger.warning("Failed to deserialize saved settings: %s. Using defaults.", e)
        
        # Build from config defaults
        return SystemSettings(
            confidence_threshold=config.CONFIDENCE_THRESHOLD,
            iou_threshold=config.IOU_THRESHOLD,
            frame_skip=config.FRAME_SKIP,
            max_processing_fps=config.MAX_PROCESSING_FPS,
            enable_signal_jump=config.ENABLE_SIGNAL_JUMP,
            enable_triple_riding=config.ENABLE_TRIPLE_RIDING,
            enable_helmet_check=config.ENABLE_HELMET_CHECK,
            enable_plate_ocr=config.ENABLE_PLATE_OCR,
            privacy_blur=config.PRIVACY_BLUR,
            calibration=CalibrationSettings(
                stop_line=config.STOP_LINE,
                signal_roi=config.SIGNAL_ROI,
                signal_rois=[config.SIGNAL_ROI],
                signal_mode=config.SIGNAL_MODE
            )
        )

    @staticmethod
    def update_settings(new_settings: SystemSettings) -> SystemSettings:
        save_setting("system_settings", new_settings.model_dump())
        logger.info("Updated system settings.")
        return new_settings
