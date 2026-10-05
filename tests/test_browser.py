"""The web UI through a real browser: login, a chat roundtrip, slash commands and prompt-box editing.

The agents are mocks (`agents:echo` answers with the prompt, `agents:summarizer` compacts to a fixed
digest), so every assertion is deterministic. Every test runs once per browser: `--browser both` is the
default (the sandbox and the guest run it); `--browser chromium` or `--browser firefox` runs one, for a
developer's own iteration.

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


async def test_slash_menu_lists_commands(page: Page, session_daemon: Paths, tmp_path: Path) -> None:
    await seed_session(session_daemon, tmp_path, "menu", "agents:echo")
    await open_session(page, "menu")
    box = page.get_by_test_id("chat-input")
    await box.fill("/")
    menu = page.get_by_test_id("slash-menu")
    await menu.wait_for()
    items = menu.get_by_test_id("slash-item")
    assert await items.count() == 2
    rendered = [await items.nth(i).text_content() for i in range(2)]
    assert any("/help" in (t or "") and "List these commands" in (t or "") for t in rendered)
    assert any("/compact" in (t or "") for t in rendered)
    await box.press("Escape")
    await menu.wait_for(state="hidden")
    assert await box.input_value() == "/"


async def test_slash_menu_completes_keyboard(page: Page, session_daemon: Paths, tmp_path: Path) -> None:
    await seed_session(session_daemon, tmp_path, "complete", "agents:echo")
    await open_session(page, "complete")
    box = page.get_by_test_id("chat-input")
    await box.fill("/c")
    menu = page.get_by_test_id("slash-menu")
    await menu.wait_for()
    assert await menu.get_by_test_id("slash-item").count() == 1
    await box.press("ArrowDown")
    await box.press("ArrowUp")
    await box.press("Enter")
    await menu.wait_for(state="hidden")
    assert await box.input_value() == "/compact "


async def test_slash_menu_completes_click(page: Page, session_daemon: Paths, tmp_path: Path) -> None:
    await seed_session(session_daemon, tmp_path, "pick", "agents:echo")
    await open_session(page, "pick")
    box = page.get_by_test_id("chat-input")
    await box.fill("/")
    menu = page.get_by_test_id("slash-menu")
    await menu.wait_for()
    await menu.get_by_test_id("slash-item").filter(has_text="/compact").click()
    await menu.wait_for(state="hidden")
    assert await box.input_value() == "/compact "
    await send_chat(page, "/compact ")
    await expect_chat_text(page, "Context compacted")


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


async def test_prompt_ctrl_u_clears_to_line_start(page: Page, session_daemon: Paths, tmp_path: Path) -> None:
    await seed_session(session_daemon, tmp_path, "undo", "agents:echo")
    await open_session(page, "undo")
    box = page.get_by_test_id("chat-input")
    await box.fill("hello world")
    for _ in range(5):
        await box.press("ArrowLeft")
    await box.press("Control+u")
    assert await box.input_value() == "world"


async def test_prompt_ctrl_u_clears_second_line(page: Page, session_daemon: Paths, tmp_path: Path) -> None:
    await seed_session(session_daemon, tmp_path, "multiline", "agents:echo")
    await open_session(page, "multiline")
    box = page.get_by_test_id("chat-input")
    await box.fill("ab\ncd")
    await box.press("Control+u")
    assert await box.input_value() == "ab\n"


async def test_prompt_ctrl_k_clears_to_line_end(page: Page, session_daemon: Paths, tmp_path: Path) -> None:
    await seed_session(session_daemon, tmp_path, "kill", "agents:echo")
    await open_session(page, "kill")
    box = page.get_by_test_id("chat-input")
    await box.fill("hello world")
    await box.press("Home")
    for _ in range(6):
        await box.press("ArrowRight")
    await box.press("Control+k")
    assert await box.input_value() == "hello "


async def test_prompt_ctrl_w_deletes_word_before(page: Page, session_daemon: Paths, tmp_path: Path) -> None:
    await seed_session(session_daemon, tmp_path, "word", "agents:echo")
    await open_session(page, "word")
    box = page.get_by_test_id("chat-input")
    await box.fill("hello world")
    await box.press("Control+w")
    assert await box.input_value() == "hello "


async def test_prompt_ctrl_e_moves_to_line_end(page: Page, session_daemon: Paths, tmp_path: Path) -> None:
    await seed_session(session_daemon, tmp_path, "end", "agents:echo")
    await open_session(page, "end")
    box = page.get_by_test_id("chat-input")
    await box.fill("hello")
    await box.press("Home")
    await box.press("Control+e")
    await box.press("x")
    assert await box.input_value() == "hellox"


async def test_prompt_ctrl_a_stays_select_all(page: Page, session_daemon: Paths, tmp_path: Path) -> None:
    await seed_session(session_daemon, tmp_path, "all", "agents:echo")
    await open_session(page, "all")
    box = page.get_by_test_id("chat-input")
    await box.fill("hello")
    await box.press("Control+a")
    await box.press("x")
    assert await box.input_value() == "x"
