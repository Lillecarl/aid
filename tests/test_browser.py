"""The web UI through a real browser: login, a chat roundtrip, and slash commands.

The agents are mocks (`agents:echo` answers with the prompt, `agents:summarizer` compacts to a fixed
digest), so every assertion is deterministic. One run tests one browser: `--browser chromium` in the Nix
sandbox, `--browser firefox` for a developer's own run.

The `page` fixture arrives logged in: the session login walks dex's form once, and each test gets an
isolated tab from its cookies. Tests seed their sessions on the shared session daemon under unique names.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import anyio
import pytest

from tests.browser_harness import expect_chat_text, open_session, seed_session, send_chat
from tests.web_harness import ALLOWED, ASSETS, TIMEOUT, needs_dex

if TYPE_CHECKING:
    from pathlib import Path

    from playwright.async_api import Page

    from aid.paths import Paths

pytestmark = [
    pytest.mark.anyio,
    needs_dex,
    pytest.mark.skipif(ASSETS is None, reason="AID_WEB_ASSETS is not set"),
]


async def test_login_shows_the_user(page: Page) -> None:
    # A fresh tab selects nothing, whatever other tests seeded: the empty prompt is a placeholder paragraph,
    # not a chat row, because no session has spoken yet.
    with anyio.fail_after(TIMEOUT):
        await page.get_by_text("Select or create a session.").wait_for()
    assert (await page.get_by_test_id("user-email").text_content()) == ALLOWED


async def test_prompt_echo_roundtrip(page: Page, session_daemon: Paths, tmp_path: Path) -> None:
    await seed_session(session_daemon, tmp_path, "echo", "agents:echo")
    await open_session(page, "echo")
    await send_chat(page, "hello browser")
    await expect_chat_text(page, "turn 1: echo hello browser")


async def test_slash_help_lists_commands(page: Page, session_daemon: Paths, tmp_path: Path) -> None:
    await seed_session(session_daemon, tmp_path, "help", "agents:echo")
    await open_session(page, "help")
    await send_chat(page, "/help")
    await expect_chat_text(page, "/help — List these commands")
    await expect_chat_text(page, "/compact [focus]")


async def test_slash_compact_replaces_history(page: Page, session_daemon: Paths, tmp_path: Path) -> None:
    await seed_session(session_daemon, tmp_path, "condense", "agents:summarizer")
    await open_session(page, "condense")
    await send_chat(page, "first")
    await expect_chat_text(page, "kept decisions")
    await send_chat(page, "second")
    await send_chat(page, "/compact")
    await expect_chat_text(page, "Context compacted")
    await expect_chat_text(page, "kept decisions")
