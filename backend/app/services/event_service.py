import uuid
import json
import logging
from datetime import datetime, timezone, timedelta
from typing import Optional, List, Dict, Any

from backend.app.database.db import get_db_connection
from backend.app.schemas.event import (
    ViolationEventCreate,
    ViolationEventUpdate,
    ViolationEventResponse,
    PaginatedEventsResponse,
    DashboardStatsResponse,
    EventType,
    ReviewStatus,
    Severity,
)

logger = logging.getLogger("traffic_system.event_service")

class EventService:
    @staticmethod
    def create_event(event_in: ViolationEventCreate) -> ViolationEventResponse:
        conn = get_db_connection()
        try:
            event_id = event_in.id or str(uuid.uuid4())
            now = datetime.now(timezone.utc).isoformat()
            details_str = json.dumps(event_in.details) if event_in.details else None

            with conn:
                conn.execute("""
                INSERT INTO violation_events (
                    id, event_type, timestamp, camera_id, tracking_id, confidence,
                    severity, status, evidence_path, crop_path, plate_text,
                    plate_raw_text, plate_confidence, details_json, reviewer_notes,
                    created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    event_id,
                    event_in.event_type.value if hasattr(event_in.event_type, 'value') else str(event_in.event_type),
                    event_in.timestamp,
                    event_in.camera_id,
                    event_in.tracking_id,
                    event_in.confidence,
                    event_in.severity.value if hasattr(event_in.severity, 'value') else str(event_in.severity),
                    event_in.status.value if hasattr(event_in.status, 'value') else str(event_in.status),
                    event_in.evidence_path,
                    event_in.crop_path,
                    event_in.plate_text,
                    event_in.plate_raw_text,
                    event_in.plate_confidence,
                    details_str,
                    event_in.reviewer_notes,
                    now,
                    now
                ))

            logger.info("Created violation event %s of type %s", event_id, event_in.event_type)
            return EventService.get_event(event_id)
        finally:
            conn.close()

    @staticmethod
    def is_duplicate(event_type: str, tracking_id: Optional[int], window_seconds: float = 4.0) -> bool:
        """Check if an event of the same type was already recorded for this tracked vehicle recently."""
        if tracking_id is None:
            return False
        conn = get_db_connection()
        try:
            # Check within the last few minutes
            cutoff = (datetime.now(timezone.utc) - timedelta(seconds=window_seconds)).isoformat()
            cursor = conn.execute("""
            SELECT id FROM violation_events
            WHERE event_type = ? AND tracking_id = ? AND created_at >= ?
            LIMIT 1
            """, (event_type, tracking_id, cutoff))
            row = cursor.fetchone()
            return row is not None
        finally:
            conn.close()

    @staticmethod
    def get_event(event_id: str) -> Optional[ViolationEventResponse]:
        conn = get_db_connection()
        try:
            cursor = conn.execute("SELECT * FROM violation_events WHERE id = ?", (event_id,))
            row = cursor.fetchone()
            if not row:
                return None
            return EventService._row_to_model(row)
        finally:
            conn.close()

    @staticmethod
    def list_events(
        event_type: Optional[str] = None,
        status: Optional[str] = None,
        search: Optional[str] = None,
        page: int = 1,
        page_size: int = 50,
    ) -> PaginatedEventsResponse:
        conn = get_db_connection()
        try:
            query = "SELECT * FROM violation_events WHERE 1=1"
            count_query = "SELECT COUNT(*) as total FROM violation_events WHERE 1=1"
            params: List[Any] = []

            if event_type:
                query += " AND event_type = ?"
                count_query += " AND event_type = ?"
                params.append(event_type)

            if status:
                query += " AND status = ?"
                count_query += " AND status = ?"
                params.append(status)

            if search:
                query += " AND (plate_text LIKE ? OR id LIKE ? OR reviewer_notes LIKE ?)"
                count_query += " AND (plate_text LIKE ? OR id LIKE ? OR reviewer_notes LIKE ?)"
                term = f"%{search}%"
                params.extend([term, term, term])

            count_cursor = conn.execute(count_query, params)
            total = count_cursor.fetchone()["total"]

            query += " ORDER BY created_at DESC LIMIT ? OFFSET ?"
            offset = (page - 1) * page_size
            page_params = params + [page_size, offset]

            cursor = conn.execute(query, page_params)
            rows = cursor.fetchall()
            items = [EventService._row_to_model(r) for r in rows]

            return PaginatedEventsResponse(
                total=total,
                page=page,
                page_size=page_size,
                items=items
            )
        finally:
            conn.close()

    @staticmethod
    def update_event(event_id: str, update_in: ViolationEventUpdate) -> Optional[ViolationEventResponse]:
        conn = get_db_connection()
        try:
            now = datetime.now(timezone.utc).isoformat()
            fields = []
            params = []

            if update_in.status is not None:
                fields.append("status = ?")
                params.append(update_in.status.value if hasattr(update_in.status, 'value') else str(update_in.status))

            if update_in.plate_text is not None:
                fields.append("plate_text = ?")
                params.append(update_in.plate_text)

            if update_in.reviewer_notes is not None:
                fields.append("reviewer_notes = ?")
                params.append(update_in.reviewer_notes)

            if not fields:
                return EventService.get_event(event_id)

            fields.append("updated_at = ?")
            params.append(now)
            params.append(event_id)

            with conn:
                cursor = conn.execute(
                    f"UPDATE violation_events SET {', '.join(fields)} WHERE id = ?",
                    params
                )
                if cursor.rowcount == 0:
                    return None

            return EventService.get_event(event_id)
        finally:
            conn.close()

    @staticmethod
    def get_statistics() -> DashboardStatsResponse:
        conn = get_db_connection()
        try:
            total_cursor = conn.execute("SELECT COUNT(*) as total FROM violation_events")
            total = total_cursor.fetchone()["total"]

            red_cursor = conn.execute("SELECT COUNT(*) as cnt FROM violation_events WHERE event_type = ?", (EventType.RED_LIGHT_CROSSING.value,))
            red_count = red_cursor.fetchone()["cnt"]

            triple_cursor = conn.execute("SELECT COUNT(*) as cnt FROM violation_events WHERE event_type = ?", (EventType.TRIPLE_RIDING.value,))
            triple_count = triple_cursor.fetchone()["cnt"]

            helmet_cursor = conn.execute("SELECT COUNT(*) as cnt FROM violation_events WHERE event_type = ?", (EventType.NO_HELMET.value,))
            helmet_count = helmet_cursor.fetchone()["cnt"]

            pending_cursor = conn.execute("SELECT COUNT(*) as cnt FROM violation_events WHERE status = ?", (ReviewStatus.PENDING_REVIEW.value,))
            pending_count = pending_cursor.fetchone()["cnt"]

            confirmed_cursor = conn.execute("SELECT COUNT(*) as cnt FROM violation_events WHERE status = ?", (ReviewStatus.CONFIRMED_BY_REVIEWER.value,))
            confirmed_count = confirmed_cursor.fetchone()["cnt"]

            dismissed_cursor = conn.execute("SELECT COUNT(*) as cnt FROM violation_events WHERE status = ?", (ReviewStatus.DISMISSED.value,))
            dismissed_count = dismissed_cursor.fetchone()["cnt"]

            recent_cursor = conn.execute("SELECT * FROM violation_events ORDER BY created_at DESC LIMIT 5")
            recent_rows = recent_cursor.fetchall()
            recent_events = [EventService._row_to_model(r) for r in recent_rows]

            return DashboardStatsResponse(
                total_violations=total,
                red_light_count=red_count,
                triple_riding_count=triple_count,
                no_helmet_count=helmet_count,
                pending_review_count=pending_count,
                confirmed_count=confirmed_count,
                dismissed_count=dismissed_count,
                recent_events=recent_events
            )
        finally:
            conn.close()

    @staticmethod
    def clear_all_events():
        """Clears all violation events from database upon explicit user confirmation."""
        conn = get_db_connection()
        try:
            with conn:
                conn.execute("DELETE FROM violation_events")
            logger.info("Cleared all violation events from database upon request.")
        finally:
            conn.close()

    @staticmethod
    def _row_to_model(row: Any) -> ViolationEventResponse:
        details = None
        if row["details_json"]:
            try:
                details = json.loads(row["details_json"])
            except Exception:
                details = None

        return ViolationEventResponse(
            id=row["id"],
            event_type=EventType(row["event_type"]),
            timestamp=row["timestamp"],
            camera_id=row["camera_id"],
            tracking_id=row["tracking_id"],
            confidence=row["confidence"],
            severity=Severity(row["severity"]),
            status=ReviewStatus(row["status"]),
            evidence_path=row["evidence_path"],
            crop_path=row["crop_path"],
            plate_text=row["plate_text"],
            plate_raw_text=row["plate_raw_text"],
            plate_confidence=row["plate_confidence"],
            details=details,
            reviewer_notes=row["reviewer_notes"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )
