"""JSON events on stderr; never log prompts, credentials, or reference answers."""

import json
import logging


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        return json.dumps(
            {
                "level": record.levelname,
                "logger": record.name,
                "event": record.getMessage(),
                **({"details": record.details} if hasattr(record, "details") else {}),
            },
            ensure_ascii=False,
        )


def configure_logging() -> logging.Logger:
    logger = logging.getLogger("multilingual_llm_reasoning")
    handler = logging.StreamHandler()
    handler.setFormatter(JsonFormatter())
    logger.handlers.clear()
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    return logger
