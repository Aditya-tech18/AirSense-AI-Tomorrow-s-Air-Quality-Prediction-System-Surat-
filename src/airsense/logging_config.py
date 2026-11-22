"""Allowlisted structured events on stdout, without request payloads."""

from datetime import datetime, timezone
import json
import logging
import sys


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        event = {
            "event": record.getMessage(),
            "timestamp": datetime.fromtimestamp(record.created, timezone.utc).isoformat(),
            "level": record.levelname,
        }
        for key in ("status", "latency_ms", "reason"):
            if hasattr(record, key):
                event[key] = getattr(record, key)
        return json.dumps(event, allow_nan=False)


def configure_logger() -> logging.Logger:
    logger = logging.getLogger("airsense.serving")
    # A repeated factory call must not accumulate duplicate stdout handlers.
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(JsonFormatter())
        logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    return logger
