from __future__ import annotations

import logging

from aid.log import configure_logging


def test_request_urls_stay_out_of_the_log() -> None:
    """Telegram puts the bot token in the URL; httpx logs URLs at info."""
    root = logging.getLogger()
    previous = root.level
    try:
        configure_logging(logging.DEBUG)
        assert not logging.getLogger("httpx").isEnabledFor(logging.INFO)
        assert logging.getLogger("httpx").isEnabledFor(logging.WARNING)
        assert logging.getLogger("aid.daemon").isEnabledFor(logging.DEBUG)
    finally:
        root.setLevel(previous)
