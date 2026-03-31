import json
import logging
import sys
import time
from contextlib import contextmanager
from typing import Any, Dict


_CONFIGURED = False


def _configure_root_logger() -> None:
    global _CONFIGURED
    if _CONFIGURED:
        return

    handler = logging.StreamHandler(stream=sys.stdout)

    class JsonFormatter(logging.Formatter):
        def format(self, record: logging.LogRecord) -> str:
            payload: Dict[str, Any] = {
                "ts": time.time(),
                "level": record.levelname,
                "logger": record.name,
                "message": record.getMessage(),
            }
            if hasattr(record, "event"):
                payload["event"] = getattr(record, "event")
            if hasattr(record, "fields"):
                payload.update(getattr(record, "fields"))
            if record.exc_info:
                payload["exception"] = self.formatException(record.exc_info)
            return json.dumps(payload, ensure_ascii=False)

    handler.setFormatter(JsonFormatter())

    root = logging.getLogger()
    root.setLevel(logging.INFO)
    root.addHandler(handler)

    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    _configure_root_logger()
    return logging.getLogger(name)


@contextmanager
def log_timed(logger: logging.Logger, event: str, **fields: Any):
    start = time.perf_counter()
    try:
        yield
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        logger.info(
            "",
            extra={
                "event": event,
                "fields": {**fields, "elapsed_ms": elapsed_ms},
            },
        )
    except Exception as exc:
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        logger.error(
            f"{exc}",
            exc_info=True,
            extra={
                "event": event,
                "fields": {**fields, "elapsed_ms": elapsed_ms},
            },
        )
        raise

