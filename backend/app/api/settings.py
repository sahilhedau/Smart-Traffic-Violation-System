from fastapi import APIRouter
from backend.app.schemas.settings import SystemSettings
from backend.app.services.settings_service import SettingsService

router = APIRouter(prefix="/settings", tags=["settings"])

@router.get("", response_model=SystemSettings)
def get_settings():
    return SettingsService.get_settings()

@router.patch("", response_model=SystemSettings)
def update_settings(settings: SystemSettings):
    return SettingsService.update_settings(settings)
