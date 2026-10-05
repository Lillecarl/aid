"""The web UI through a real browser: login, a chat roundtrip, slash commands, prompt-box editing, PWA
installability, thinking visibility and cost display.

The agents are mocks (`agents:echo` answers with the prompt, `agents:summarizer` compacts to a fixed
digest), so every assertion is deterministic. Every test runs once per browser: `--browser both` is the
default (the sandbox and the guest run it); `--browser chromium` or `--browser firefox` runs one, for a
developer's own iteration.

The `page` fixture arrives logged in: the session login walks dex's form once, and each test gets an
isolated tab from its cookies. Tests seed their sessions on the shared session daemon under unique names.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import anyio
import pytest

import aid
from tests.browser_harness import (
    expect_chat_text,
    open_session,
    seed_acp_session,
    seed_session,
    send_chat,
    wait_for_idle,
)
from tests.test_agents import AGENT_DIR
from tests.web_harness import ALLOWED, ASSETS, TIMEOUT, needs_dex

if TYPE_CHECKING:
    from collections.abc import AsyncIterator
    from pathlib import Path

    from playwright.async_api import Page

    from aid.paths import Paths

pytestmark = [
    pytest.mark.anyio,
    needs_dex,
    pytest.mark.skipif(ASSETS is None, reason="AID_WEB_ASSETS is not set"),
]


@pytest.fixture(autouse=True)
async def clean_sessions(session_daemon: Paths) -> AsyncIterator[None]:
    """Each test deletes its sessions when done: their workers exit, so the next test starts near-empty
    instead of accumulating a worker per session until the guest falls over."""
    yield
    async with aid.connect(session_daemon) as client:
        for session in await client.sessions():
            await client.session(session.name).delete()


async def test_pwa_is_installable(page: Page, browser_name: str) -> None:
    if browser_name != "chromium":
        pytest.skip("installability is a chromium concept")
    assert await page.locator('link[rel="manifest"]').count() >= 1
    cdp = await page.context.new_cdp_session(page)
    # The CDP bridge is untyped; the verdict below is a plain JSON object.
    result = cast("dict[str, object]", await cdp.send("Page.getInstallabilityErrors"))  # pyright: ignore[reportUnknownMemberType]
    assert result.get("installabilityErrors", []) == []


async def test_login_shows_the_user(page: Page) -> None:
    # A fresh tab selects nothing, whatever other tests seeded: the empty prompt is a placeholder paragraph,
    # not a chat row, because no session has spoken yet.
    with anyio.fail_after(TIMEOUT):
        await page.get_by_text("Select or create a session.").wait_for(timeout=TIMEOUT * 1000)
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
    await menu.wait_for(timeout=TIMEOUT * 1000)
    items = menu.get_by_test_id("slash-item")
    assert await items.count() == 2
    rendered = [await items.nth(i).text_content() for i in range(2)]
    assert any("/help" in (t or "") and "List these commands" in (t or "") for t in rendered)
    assert any("/compact" in (t or "") for t in rendered)
    await box.press("Escape")
    await menu.wait_for(state="hidden", timeout=TIMEOUT * 1000)
    assert await box.input_value() == "/"


async def test_slash_menu_completes_keyboard(page: Page, session_daemon: Paths, tmp_path: Path) -> None:
    await seed_session(session_daemon, tmp_path, "complete", "agents:echo")
    await open_session(page, "complete")
    box = page.get_by_test_id("chat-input")
    await box.fill("/c")
    menu = page.get_by_test_id("slash-menu")
    await menu.wait_for(timeout=TIMEOUT * 1000)
    assert await menu.get_by_test_id("slash-item").count() == 1
    await box.press("ArrowDown")
    await box.press("ArrowUp")
    await box.press("Enter")
    await menu.wait_for(state="hidden", timeout=TIMEOUT * 1000)
    assert await box.input_value() == "/compact "


async def test_slash_menu_completes_click(page: Page, session_daemon: Paths, tmp_path: Path) -> None:
    await seed_session(session_daemon, tmp_path, "pick", "agents:echo")
    await open_session(page, "pick")
    box = page.get_by_test_id("chat-input")
    await box.fill("/")
    menu = page.get_by_test_id("slash-menu")
    await menu.wait_for(timeout=TIMEOUT * 1000)
    await menu.get_by_test_id("slash-item").filter(has_text="/compact").click()
    await menu.wait_for(state="hidden", timeout=TIMEOUT * 1000)
    assert await box.input_value() == "/compact "
    await send_chat(page, "/compact ")
    await expect_chat_text(page, "Context compacted")


async def test_slash_enter_runs_complete_command(page: Page, session_daemon: Paths, tmp_path: Path) -> None:
    await seed_session(session_daemon, tmp_path, "exact", "agents:echo")
    await open_session(page, "exact")
    box = page.get_by_test_id("chat-input")
    # Whether or not the menu opened before Enter arrived, a complete command sends: it never completes.
    await box.fill("/help")
    await box.press("Enter")
    await expect_chat_text(page, "/help — List these commands")
    await expect_chat_text(page, "/compact [focus]")


async def test_slash_help_lists_commands(page: Page, session_daemon: Paths, tmp_path: Path) -> None:
    await seed_session(session_daemon, tmp_path, "help", "agents:echo")
    await open_session(page, "help")
    await send_chat(page, "/help")
    await expect_chat_text(page, "/help — List these commands")
    await expect_chat_text(page, "/compact [focus]")
    await page.locator("details.help[open]").wait_for(timeout=TIMEOUT * 1000)


async def test_slash_help_stays_out_of_history(page: Page, session_daemon: Paths, tmp_path: Path) -> None:
    await seed_session(session_daemon, tmp_path, "untainted", "agents:echo")
    await open_session(page, "untainted")
    await send_chat(page, "/help")
    await expect_chat_text(page, "/help — List these commands")
    async with aid.connect(session_daemon) as client:
        entries = (await client.session("untainted").history()).entries
    assert all("List these commands" not in str(entry) for entry in entries)


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


async def test_tool_calls_share_a_row(page: Page, session_daemon: Paths, tmp_path: Path) -> None:
    await seed_session(session_daemon, tmp_path, "tooling", "agents:tool_caller", AID_AGENTS_PATH=str(AGENT_DIR))
    await open_session(page, "tooling")
    await send_chat(page, "go")
    boxes = page.get_by_test_id("tool-box")
    await boxes.nth(1).wait_for(timeout=TIMEOUT * 1000)
    await wait_for_idle(page)
    flows = page.get_by_test_id("flow-row")
    await flows.first.wait_for(timeout=TIMEOUT * 1000)
    assert await flows.first.get_by_test_id("tool-box").count() >= 2
    first = await boxes.nth(0).bounding_box()
    second = await boxes.nth(1).bounding_box()
    assert first is not None and second is not None
    # One flex row: the vertical ranges overlap.
    assert first["y"] < second["y"] + second["height"] and second["y"] < first["y"] + first["height"]
    rows = page.get_by_test_id("chat-row")
    last = await rows.last.bounding_box()
    assert last is not None
    # An agent message breaks the flow: the last row starts below the tool row.
    assert last["y"] >= max(first["y"] + first["height"], second["y"] + second["height"])


async def test_status_shows_accumulated_cost(page: Page, session_daemon: Paths, tmp_path: Path) -> None:
    await seed_acp_session(session_daemon, tmp_path, "spender")
    await open_session(page, "spender")
    await send_chat(page, "usage")
    await expect_chat_text(page, "0.50 USD")
    # Same agent session twice: the latest figure counts once, it never doubles.
    await send_chat(page, "usage")
    await wait_for_idle(page)
    await page.get_by_role("tab", name="Status").click()
    cost = page.get_by_test_id("status-cost")
    await cost.wait_for(timeout=TIMEOUT * 1000)
    assert await cost.text_content() == "0.50 USD"


async def test_thinking_visible_until_done(page: Page, session_daemon: Paths, tmp_path: Path) -> None:
    gate = tmp_path / "think-gate"
    await seed_session(session_daemon, tmp_path, "ponder", "agents:thinker", AID_THINK_GATE=str(gate))
    await open_session(page, "ponder")
    await send_chat(page, "think")
    # The turn cannot finish before the gate file exists: the open thought stays put for the assertion.
    streaming = page.locator("details.thought[open]")
    await streaming.wait_for(timeout=TIMEOUT * 1000)
    assert "hmm" in (await streaming.text_content() or "")
    # Deltas join one box per turn instead of spawning one each.
    assert await page.locator("details.thought").count() == 1
    await anyio.Path(gate).write_text("go")
    await wait_for_idle(page)
    await page.locator("details.thought:not([open])").wait_for(timeout=TIMEOUT * 1000)
    assert await page.locator("details.thought[open]").count() == 0
    assert await page.locator("details.thought").count() == 1
    assert "let me think" in (await page.locator("details.thought").text_content() or "")


async def test_thinking_collapsed_mode(page: Page, session_daemon: Paths, tmp_path: Path) -> None:
    gate = tmp_path / "think-gate"
    await seed_session(session_daemon, tmp_path, "quiesce", "agents:thinker", AID_THINK_GATE=str(gate))
    await open_session(page, "quiesce")
    await page.get_by_label("Thinking").select_option("collapsed")
    await send_chat(page, "think")
    await page.locator("details.thought").wait_for(timeout=TIMEOUT * 1000)
    await anyio.Path(gate).write_text("go")
    await wait_for_idle(page)
    assert await page.locator("details.thought").count() >= 1
    assert await page.locator("details.thought[open]").count() == 0


async def test_thinking_expands_on_demand(page: Page, session_daemon: Paths, tmp_path: Path) -> None:
    gate = tmp_path / "think-gate"
    await seed_session(session_daemon, tmp_path, "unfold", "agents:thinker", AID_THINK_GATE=str(gate))
    await open_session(page, "unfold")
    await send_chat(page, "think")
    await page.locator("details.thought").wait_for(timeout=TIMEOUT * 1000)
    await anyio.Path(gate).write_text("go")
    await wait_for_idle(page)
    await page.locator("details.thought:not([open])").wait_for(timeout=TIMEOUT * 1000)
    await page.locator("details.thought summary").click()
    await page.locator("details.thought[open]").wait_for(timeout=TIMEOUT * 1000)
