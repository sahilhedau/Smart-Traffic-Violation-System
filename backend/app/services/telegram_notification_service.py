import logging
import mimetypes
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import httpx
from PIL import Image, UnidentifiedImageError

from backend.app.config import config
from backend.app.schemas.event import EventType, ReviewStatus
from backend.app.services.event_service import EventService

logger = logging.getLogger("traffic_system.telegram")

VIOLATION_LABELS = {
    EventType.RED_LIGHT_CROSSING: "Red Light Crossing",
    EventType.TRIPLE_RIDING: "Triple Riding",
    EventType.NO_HELMET: "No Helmet",
}


class TelegramNotificationService:
    @staticmethod
    def _credentials() -> tuple[Optional[str], Optional[str]]:
        token = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
        chat_id = os.getenv("TELEGRAM_CHAT_ID", "").strip()
        return token or None, chat_id or None

    @staticmethod
    def _request(
        method: str,
        token: str,
        chat_id: str,
        text: str,
        evidence_path: Optional[str] = None
    ) -> tuple[bool, Optional[str]]:
        url = f"https://api.telegram.org/bot{token}/{method}"
        timeout = httpx.Timeout(10.0, connect=3.0)

        try:
            if evidence_path:
                path = Path(evidence_path)
                mime_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
                with path.open("rb") as image:
                    response = httpx.post(
                        url,
                        data={
                            "chat_id": chat_id,
                            "caption": text,
                        },
                        files={
                            "photo": (path.name, image, mime_type),
                        },
                        timeout=timeout,
                    )
            else:
                response = httpx.post(
                    url,
                    data={"chat_id": chat_id, "text": text},
                    timeout=timeout,
                )
        except httpx.TimeoutException:
            return False, "Telegram request timed out"
        except httpx.RequestError as error:
            logger.warning("Telegram request failed (%s)", type(error).__name__)
            return False, "Telegram network request failed"
        except OSError:
            return False, "Evidence image could not be read"

        if response.status_code != 200:
            return False, f"Telegram API returned HTTP {response.status_code}"

        try:
            result = response.json()
        except ValueError:
            return False, "Telegram returned an invalid response"
        if not isinstance(result, dict) or result.get("ok") is not True:
            error_code = result.get("error_code") if isinstance(result, dict) else None
            if error_code == 401:
                return False, "Telegram rejected the bot token"
            if error_code == 400:
                return False, "Telegram rejected the chat or message"
            return False, "Telegram could not send the notification"
        return True, None

    @staticmethod
    def _valid_evidence_path(evidence_path: Optional[str]) -> Optional[str]:
        if not evidence_path:
            return None
        path = Path(evidence_path)
        if not path.is_file():
            return None
        try:
            with Image.open(path) as evidence:
                evidence.verify()
        except (OSError, UnidentifiedImageError, ValueError):
            return None
        return str(path)

    @staticmethod
    def _event_message(event) -> str:
        try:
            event_type = EventType(event.event_type)
        except ValueError:
            label = str(event.event_type)
        else:
            label = VIOLATION_LABELS.get(event_type, event_type.value)

        lines = [
            "TrafficGuard AI",
            f"Violation Type: {label}",
            f"Event ID: {event.id}",
            f"Timestamp: {event.timestamp}",
        ]
        if event.tracking_id is not None:
            lines.append(f"Tracking ID: {event.tracking_id}")
        if event.plate_text and event.plate_text.upper() != "UNREADABLE":
            lines.append(f"License Plate OCR: {event.plate_text}")
        lines.extend([
            f"AI Confidence: {event.confidence * 100:.1f}%",
            "Review Status: CONFIRMED BY REVIEWER",
        ])
        return "\n".join(lines)

    @classmethod
    def send_test_message(cls) -> tuple[bool, str]:
        token, chat_id = cls._credentials()
        if not token or not chat_id:
            return False, "Telegram is not configured (set TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID)"
        sent, error = cls._request(
            "sendMessage",
            token,
            chat_id,
            "TrafficGuard AI\nTEST MESSAGE: Telegram notifications are configured.",
        )
        return sent, "Test message sent" if sent else (error or "Telegram send failed")

    @classmethod
    def notify_confirmed_event(
        cls,
        event_id: str,
        retry: bool = False
    ) -> tuple[bool, str]:
        event = EventService.get_event(event_id)
        if event is None:
            return False, "Violation event not found"
        if event.status != ReviewStatus.CONFIRMED_BY_REVIEWER:
            return False, "Only confirmed violations can be sent"
        if not EventService.claim_telegram_notification(event_id, retry=retry):
            return False, "Notification already sent or currently being processed"

        token, chat_id = cls._credentials()
        if not token or not chat_id:
            cls._record_status(event_id, "NOT_CONFIGURED", "Telegram is not configured")
            return False, "Telegram is not configured"

        evidence_path = cls._valid_evidence_path(event.evidence_path)
        sent, error = cls._request(
            "sendPhoto" if evidence_path else "sendMessage",
            token,
            chat_id,
            cls._event_message(event),
            evidence_path,
        )
        if not sent and evidence_path and error == "Evidence image could not be read":
            sent, error = cls._request(
                "sendMessage",
                token,
                chat_id,
                cls._event_message(event),
            )
        if sent:
            cls._record_status(
                event_id,
                "SENT",
                None,
                datetime.now(timezone.utc).isoformat(),
            )
            return True, "Telegram notification sent"

        cls._record_status(event_id, "FAILED", error or "Telegram send failed")
        logger.warning("Telegram notification failed for event %s: %s", event_id, error)
        return False, error or "Telegram send failed"

    @staticmethod
    def _record_status(
        event_id: str,
        status: str,
        error: Optional[str],
        sent_at: Optional[str] = None
    ) -> None:
        try:
            EventService.update_telegram_notification(
                event_id,
                status,
                error,
                sent_at,
            )
        except Exception:
            logger.exception(
                "Could not persist Telegram notification status for event %s",
                event_id
            )
