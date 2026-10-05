"""Structured logging for every aid process.

Each module logs through `structlog.get_logger(__name__)` with an event and keys:

    log.info("worker_ready", session=name, pid=pid)

never a %-formatted string. `configure_logging` installs the pipeline once per
process: JSON rendering when `AID_LOG_JSON` is set, a console renderer otherwise.
The pipeline sits on stdlib handlers, so third-party logs (hypercorn, watchfiles)
flow through the same formatter. Logger caching stays off: forkserver workers
inherit the daemon's memory, and a cached logger would keep its pipeline.
"""

from __future__ import annotations

import logging
import os
import sys
from typing import TYPE_CHECKING

import structlog

from aid.paths import ENV_LOG_JSON

if TYPE_CHECKING:
    from structlog.typing import Processor


def configure_logging(level: int = logging.INFO) -> None:
    """Install the structlog pipeline on the root stdlib handler. Idempotent."""
    shared: list[Processor] = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso", utc=True),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
        structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
    ]
    renderer = structlog.processors.JSONRenderer() if os.environ.get(ENV_LOG_JSON) else structlog.dev.ConsoleRenderer()
    formatter = structlog.stdlib.ProcessorFormatter(
        processors=[structlog.stdlib.ProcessorFormatter.remove_processors_meta, renderer],
        foreign_pre_chain=[
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.processors.TimeStamper(fmt="iso", utc=True),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
        ],
    )
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(formatter)
    root = logging.getLogger()
    if not any(isinstance(h.formatter, structlog.stdlib.ProcessorFormatter) for h in root.handlers):
        root.addHandler(handler)
    root.setLevel(level)
    structlog.configure(
        processors=shared,
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=False,
    )
