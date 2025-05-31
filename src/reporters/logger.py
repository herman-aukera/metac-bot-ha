"""
Structured logging for forecasts.
"""
import structlog
import logging

logging.basicConfig(level=logging.INFO)
structlog.configure(
    processors=[
        structlog.processors.KeyValueRenderer(key_order=["timestamp", "level", "event", "agent"])
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    wrapper_class=structlog.stdlib.BoundLogger,
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger()
