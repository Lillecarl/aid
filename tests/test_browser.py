"""The web UI through a real browser: login, a chat roundtrip, and slash commands.

The agents are mocks (`agents:echo` answers with the prompt, `agents:summarizer` compacts to a fixed
digest), so every assertion is deterministic. One run tests one browser: `--browser chromium` in the Nix
sandbox, `--browser firefox` for a developer's own run.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from tests.browser_harness import expect_chat_text, login_ui, open_session, seed_session, send_chat
from tests.web_harness import ALLOWED, ASSETS, needs_dex

if TYPE_CHECKING:
    from pathlib import Path

    from playwright.async_api import Page

    from aid.paths import Paths
    from tests.web_harness import Web

pytestmark = [
    pytest.mark.anyio,
    needs_dex,
    pytest.mark.skipif(ASSETS is None, reason="AID_WEB_ASSETS is not set"),
]


async def test_login_shows_the_user(page: Page, web: Web) -> None:
    await login_ui(page, web)
    await expect_chat_text(page, "Select or create a session.")
    assert (await page.get_by_test_id("user-email").text_content()) == ALLOWED


async def test_prompt_echo_roundtrip(page: Page, web: Web, daemon: Paths, tmp_path: Path) -> None:
    await seed_session(daemon, tmp_path, "echo", "agents:echo")
    await login_ui(page, web)
    await open_session(page, "echo")
    await send_chat(page, "hello browser")
    await expect_chat_text(page, "turn 1: echo hello browser")


async def test_slash_help_lists_commands(page: Page, web: Web, daemon: Paths, tmp_path: Path) -> None:
    await seed_session(daemon, tmp_path, "echo", "agents:echo")
    await login_ui(page, web)
    await open_session(page, "echo")
    await send_chat(page, "/help")
    await expect_chat_text(page, "/help — List these commands")
    await expect_chat_text(page, "/compact [focus]")


async def test_slash_compact_replaces_history(page: Page, web: Web, daemon: Paths, tmp_path: Path) -> None:
    await seed_session(daemon, tmp_path, "condense", "agents:summarizer")
    await login_ui(page, web)
    await open_session(page, "condense")
    await send_chat(page, "first")
    await expect_chat_text(page, "kept decisions")
    await send_chat(page, "second")
    await send_chat(page, "/compact")
    await expect_chat_text(page, "Context compacted")
    await expect_chat_text(page, "kept decisions")
