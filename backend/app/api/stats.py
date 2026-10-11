from fastapi import APIRouter
from backend.app.schemas.event import DashboardStatsResponse
from backend.app.services.event_service import EventService

router = APIRouter(prefix="/statistics", tags=["statistics"])

@router.get("", response_model=DashboardStatsResponse)
def get_dashboard_statistics():
    return EventService.get_statistics()
