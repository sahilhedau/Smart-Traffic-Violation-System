import os
import sys
from pathlib import Path
import urllib.request
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("download_models")

BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# Official verified YOLO models
MODELS = {
    "yolov8n.pt": {
        "url": "https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8n.pt",
        "description": "YOLOv8 Nano pretrained general object detector (COCO dataset)",
        "expected_min_bytes": 5_000_000,
    }
}

def download_model(filename: str, info: dict) -> bool:
    target_path = MODELS_DIR / filename
    temp_path = MODELS_DIR / f"{filename}.tmp"

    if target_path.exists() and target_path.stat().st_size >= info["expected_min_bytes"]:
        logger.info("Model '%s' already exists and is verified (%d bytes).", filename, target_path.stat().st_size)
        return True

    url = info["url"]
    logger.info("Downloading %s from %s ...", filename, url)

    try:
        def reporthook(blocknum, blocksize, totalsize):
            if totalsize > 0:
                percent = min(100.0, blocknum * blocksize / totalsize * 100.0)
                sys.stdout.write(f"\r  -> Progress: {percent:.1f}%")
                sys.stdout.flush()

        urllib.request.urlretrieve(url, str(temp_path), reporthook=reporthook)
        print()

        if temp_path.stat().st_size < info["expected_min_bytes"]:
            logger.error("Downloaded file %s is smaller than expected (%d bytes).", filename, temp_path.stat().st_size)
            if temp_path.exists():
                temp_path.unlink()
            return False

        temp_path.replace(target_path)
        logger.info("Successfully downloaded and verified %s (%d bytes).", filename, target_path.stat().st_size)
        return True
    except Exception as e:
        logger.error("Failed to download %s: %s", filename, e)
        if temp_path.exists():
            temp_path.unlink()
        return False

def main():
    logger.info("Starting model download check...")
    all_ok = True
    for filename, info in MODELS.items():
        ok = download_model(filename, info)
        if not ok:
            all_ok = False

    if all_ok:
        logger.info("All required models are ready in %s", MODELS_DIR)
    else:
        logger.warning("Some models could not be downloaded. Local fallbacks will be used where available.")

if __name__ == "__main__":
    main()
