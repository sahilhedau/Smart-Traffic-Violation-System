import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.app.database.db import init_db
from backend.app.services.telegram_notification_service import TelegramNotificationService


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Send a labeled Telegram test message or retry a confirmed event."
    )
    parser.add_argument(
        "--retry-event",
        help="Retry a failed or unconfigured notification for this confirmed event ID.",
    )
    args = parser.parse_args()

    if args.retry_event:
        init_db()
        sent, message = TelegramNotificationService.notify_confirmed_event(
            args.retry_event,
            retry=True,
        )
    else:
        sent, message = TelegramNotificationService.send_test_message()

    print(message)
    return 0 if sent else 1


if __name__ == "__main__":
    raise SystemExit(main())
