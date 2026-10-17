import pytest
from pydantic import ValidationError

from backend.app.schemas.settings import CalibrationSettings


def test_single_signal_roi_settings_are_migrated():
    calibration = CalibrationSettings(
        stop_line=[0.0, 0.5, 1.0, 0.5],
        signal_roi=[0.1, 0.2, 0.3, 0.4]
    )

    assert calibration.signal_rois == [[0.1, 0.2, 0.3, 0.4]]


def test_multiple_signal_rois_preserve_legacy_primary_roi():
    calibration = CalibrationSettings(
        stop_line=[0.0, 0.5, 1.0, 0.5],
        signal_rois=[[0.1, 0.2, 0.3, 0.4], [0.5, 0.2, 0.7, 0.4]]
    )

    assert calibration.signal_roi == calibration.signal_rois[0]
    assert len(calibration.signal_rois) == 2
    assert calibration.stop_lines == [calibration.stop_line, calibration.stop_line]


def test_multiple_stop_lines_are_paired_with_signal_boxes():
    signal_rois = [[0.1, 0.2, 0.3, 0.4], [0.5, 0.2, 0.7, 0.4]]
    stop_lines = [[0.1, 0.5, 0.4, 0.5], [0.6, 0.5, 0.9, 0.5]]

    calibration = CalibrationSettings(signal_rois=signal_rois, stop_lines=stop_lines)

    assert calibration.stop_lines == stop_lines
    assert calibration.stop_line == stop_lines[0]


def test_stop_line_count_must_match_signal_box_count():
    with pytest.raises(ValidationError):
        CalibrationSettings(
            signal_rois=[[0.1, 0.2, 0.3, 0.4], [0.5, 0.2, 0.7, 0.4]],
            stop_lines=[[0.1, 0.5, 0.4, 0.5]]
        )


def test_signal_roi_must_have_positive_area():
    with pytest.raises(ValidationError):
        CalibrationSettings(signal_rois=[[0.5, 0.2, 0.4, 0.4]])
