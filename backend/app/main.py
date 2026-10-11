import os
import torch
import logging
from pathlib import Path
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from backend.app.config import config
from backend.app.database.db import init_db
from backend.app.api.events import router as events_router
from backend.app.api.video import router as video_router
from backend.app.api.settings import router as settings_router
from backend.app.api.stats import router as stats_router

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s"
)
logger = logging.getLogger("traffic_system")

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: ensure directories and database initialized
    logger.info("Initializing Smart Traffic Violation Detection System...")
    config.ensure_directories()
    init_db()
    logger.info("Database initialized at %s", config.DATABASE_PATH)
    yield
    # Shutdown
    logger.info("System shutting down.")

app = FastAPI(
    title="Smart Traffic Violation Detection System",
    description="Automated AI-powered computer vision system for traffic violations (Signal jump, Triple riding, No Helmet, Plate OCR)",
    version="1.0.0",
    lifespan=lifespan
)

# Enable CORS for dashboard interactions
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API Routers
app.include_router(events_router, prefix="/api")
app.include_router(video_router, prefix="/api")
app.include_router(settings_router, prefix="/api")
app.include_router(stats_router, prefix="/api")

@app.get("/api/health", tags=["system"])
def health_check():
    cuda_available = torch.cuda.is_available()
    device_name = torch.cuda.get_device_name(0) if cuda_available else "CPU"
    model_path = config.MODELS_DIR / config.YOLO_MODEL_NAME

    return {
        "status": "healthy",
        "system": "Smart Traffic Violation Detection System",
        "database": "connected",
        "device": device_name,
        "cuda_available": cuda_available,
        "model_loaded": model_path.exists(),
        "model_name": config.YOLO_MODEL_NAME,
        "storage": {
            "evidence_dir": str(config.EVIDENCE_DIR),
            "annotated_videos_dir": str(config.ANNOTATED_VIDEOS_DIR),
        }
    }

# Frontend static files mounting
FRONTEND_DIR = config.BASE_DIR / "frontend"
if FRONTEND_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")

@app.get("/", tags=["ui"])
def serve_dashboard():
    index_file = FRONTEND_DIR / "index.html"
    if index_file.exists():
        return FileResponse(index_file)
    return {"message": "Smart Traffic Violation System API is operational. Frontend index.html not found."}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.app.main:app", host=config.HOST, port=config.PORT, reload=config.DEBUG)
