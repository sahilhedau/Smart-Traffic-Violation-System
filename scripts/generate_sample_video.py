import cv2
import numpy as np
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("generate_sample_video")

BASE_DIR = Path(__file__).resolve().parent.parent
SAMPLES_DIR = BASE_DIR / "data" / "samples"
SAMPLES_DIR.mkdir(parents=True, exist_ok=True)

def draw_vehicle(frame, x, y, w, h, color, label="CAR", plate_text=""):
    """Draws a vehicle with chassis, windows, wheels, and a license plate."""
    # Chassis
    cv2.rectangle(frame, (x, y), (x + w, y + h), color, -1)
    cv2.rectangle(frame, (x, y), (x + w, y + h), (30, 30, 30), 2)

    # Windshield / Roof
    cv2.rectangle(frame, (x + int(w * 0.15), y + int(h * 0.15)),
                  (x + int(w * 0.85), y + int(h * 0.55)), (180, 210, 230), -1)

    # Headlights
    cv2.circle(frame, (x + 10, y + h - 10), 6, (0, 255, 255), -1)
    cv2.circle(frame, (x + w - 10, y + h - 10), 6, (0, 255, 255), -1)

    # License Plate
    if plate_text:
        pw, ph = int(w * 0.55), 22
        px = x + (w - pw) // 2
        py = y + h - 26
        cv2.rectangle(frame, (px, py), (px + pw, py + ph), (255, 255, 255), -1)
        cv2.rectangle(frame, (px, py), (px + pw, py + ph), (0, 0, 0), 1)
        cv2.putText(frame, plate_text, (px + 4, py + 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 2)

    # Label on vehicle top
    cv2.putText(frame, label, (x + 5, max(15, y - 5)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

def draw_motorcycle_with_riders(frame, mx, my, mw, mh, num_riders=1, helmet=True, plate_text=""):
    """Draws a motorcycle with wheels and specified number of seated riders."""
    # Motorcycle Body
    cv2.rectangle(frame, (mx, my + int(mh * 0.4)), (mx + mw, my + mh), (40, 40, 180), -1)
    # Wheels
    cv2.circle(frame, (mx + 15, my + mh - 5), 12, (20, 20, 20), -1)
    cv2.circle(frame, (mx + mw - 15, my + mh - 5), 12, (20, 20, 20), -1)

    # Riders
    rider_spacing = int(mw / (num_riders + 1))
    for i in range(num_riders):
        rx = mx + (i + 1) * rider_spacing
        # Torso
        cv2.rectangle(frame, (rx - 10, my - 5), (rx + 10, my + int(mh * 0.45)), (50, 120, 200), -1)
        # Head / Helmet
        head_y = my - 20
        if helmet:
            # Helmet (bright yellow/red shell)
            cv2.circle(frame, (rx, head_y), 11, (0, 215, 255), -1)
            cv2.circle(frame, (rx, head_y), 11, (0, 0, 0), 1)
        else:
            # No helmet (skin tone + hair)
            cv2.circle(frame, (rx, head_y), 10, (140, 180, 230), -1) # skin tone
            cv2.ellipse(frame, (rx, head_y - 4), (9, 6), 0, 180, 360, (20, 20, 20), -1) # black hair

    # Plate
    if plate_text:
        pw, ph = int(mw * 0.6), 18
        px = mx + (mw - pw) // 2
        py = my + mh - 12
        cv2.rectangle(frame, (px, py), (px + pw, py + ph), (255, 255, 255), -1)
        cv2.putText(frame, plate_text, (px + 2, py + 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 0, 0), 1)

def generate_sample_video(output_filename: str = "sample_traffic.mp4", duration_sec: int = 8, fps: int = 25):
    out_path = SAMPLES_DIR / output_filename
    logger.info("Generating demonstration video at %s (%d sec, %d fps)...", out_path, duration_sec, fps)

    width, height = 1280, 720
    total_frames = duration_sec * fps
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(out_path), fourcc, fps, (width, height))

    # Stop line at y = 470
    stop_line_y = 470

    for f in range(total_frames):
        # Background: asphalt road with lanes
        frame = np.full((height, width, 3), (60, 60, 60), dtype=np.uint8)

        # Sidewalks / Green grass borders
        cv2.rectangle(frame, (0, 0), (120, height), (40, 120, 40), -1)
        cv2.rectangle(frame, (width - 120, 0), (width, height), (40, 120, 40), -1)

        # Road boundaries
        cv2.line(frame, (120, 0), (120, height), (255, 255, 255), 4)
        cv2.line(frame, (width - 120, 0), (width - 120, height), (255, 255, 255), 4)

        # Dashed lane divider
        for dy in range(0, height, 50):
            cv2.line(frame, (width // 2, dy), (width // 2, dy + 25), (255, 255, 255), 3)

        # Stop Line (Road marking)
        cv2.line(frame, (130, stop_line_y), (width - 130, stop_line_y), (255, 255, 255), 6)
        cv2.putText(frame, "STOP LINE", (150, stop_line_y - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # Traffic Signal Post & Lights (top right ROI: x: 960 to 1210, y: 36 to 216)
        signal_box_x1, signal_box_y1 = 1020, 40
        signal_box_x2, signal_box_y2 = 1100, 200
        cv2.rectangle(frame, (signal_box_x1, signal_box_y1), (signal_box_x2, signal_box_y2), (20, 20, 20), -1)
        cv2.rectangle(frame, (signal_box_x1, signal_box_y1), (signal_box_x2, signal_box_y2), (180, 180, 180), 2)

        # Traffic light turns RED during frames 40 to 180 (seconds 1.6 to 7.2)
        is_red = (f >= 30)
        red_color = (0, 0, 255) if is_red else (0, 0, 70)
        green_color = (0, 255, 0) if not is_red else (0, 70, 0)
        yellow_color = (0, 70, 70)

        # Draw red, yellow, green circles
        cv2.circle(frame, (1060, 75), 18, red_color, -1)
        cv2.circle(frame, (1060, 120), 18, yellow_color, -1)
        cv2.circle(frame, (1060, 165), 18, green_color, -1)

        # --- Dynamic Actors ---
        # 1. Car A (Red-light jumping violator):
        # Approaches from y=200 downwards and crosses stop line at y=470 around frame 60-90 while signal is RED!
        car_y = 220 + int(f * 3.5)
        car_x = 260
        draw_vehicle(frame, car_x, car_y, 110, 140, (180, 40, 40), label="CAR (Violator)", plate_text="MH12AB1234")

        # 2. Motorcycle B (Triple Riding + No Helmet violator):
        # Moves down left-middle lane with 3 riders, no helmet
        moto_y = 150 + int(f * 2.8)
        moto_x = 440
        draw_motorcycle_with_riders(frame, moto_x, moto_y, 80, 100, num_riders=3, helmet=False, plate_text="DL04C1111")

        # 3. Car C (Law-abiding vehicle):
        # Stops before the stop line at y=420 when red
        car2_target_y = 360
        car2_y = min(car2_target_y, 180 + int(f * 2.2))
        car2_x = 680
        draw_vehicle(frame, car2_x, car2_y, 110, 130, (40, 130, 200), label="CAR (Compliant)", plate_text="KA01MJ5050")

        # 4. Motorcycle D (Compliant: 1 rider, with helmet):
        moto2_y = min(370, 140 + int(f * 2.0))
        moto2_x = 830
        draw_motorcycle_with_riders(frame, moto2_x, moto2_y, 75, 95, num_riders=1, helmet=True, plate_text="HR26DQ5551")

        # Frame counter & simulation watermarks
        cv2.putText(frame, f"Simulated Traffic Feed - Frame {f+1}/{total_frames}", (140, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (240, 240, 240), 2)

        out.write(frame)

    out.release()
    logger.info("Sample video generated successfully at %s (%d frames).", out_path, total_frames)
    return str(out_path)

if __name__ == "__main__":
    generate_sample_video()
