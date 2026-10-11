import pytest
from pathlib import Path
from backend.app.config import config, SystemConfig

def test_config_paths():
    assert config.BASE_DIR.exists()
    assert config.MODELS_DIR.exists()
    assert config.OUTPUTS_DIR.exists()
    assert config.EVIDENCE_DIR.exists()
    assert config.ANNOTATED_VIDEOS_DIR.exists()
    assert config.UPLOADS_DIR.exists()
    assert config.SAMPLES_DIR.exists()

def test_config_defaults():
    assert config.CONFIDENCE_THRESHOLD > 0.0
    assert config.FRAME_SKIP >= 1
    assert len(config.STOP_LINE) == 4
    assert len(config.SIGNAL_ROI) == 4
    assert config.ENABLE_SIGNAL_JUMP is True
    assert config.ENABLE_TRIPLE_RIDING is True
    assert config.ENABLE_HELMET_CHECK is True
