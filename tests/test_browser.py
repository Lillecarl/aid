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

import json
from pathlib import Path
from typing import TYPE_CHECKING, cast

import anyio
import pytest
from pydantic import JsonValue

import aid
from aid.plugins import ENV_PLUGINS_PATH
from aid.protocol import Cost, Usage
from aid.spec import PermissionMode, Sandbox
from tests.browser_harness import (
    expect_chat_text,
    open_details,
    open_session,
    seed_acp_session,
    seed_session,
    send_chat,
    wait_for_idle,
)
from tests.conftest import py_spec
from tests.test_agents import AGENT_DIR
from tests.test_sandbox import needs_bwrap
from tests.web_harness import ALLOWED, ASSETS, TIMEOUT, needs_dex

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from playwright.async_api import Locator, Page

    from aid.paths import Paths

pytestmark = [
    pytest.mark.anyio,
    # One worker under xdist: the run's one browser and one dex login, as the harness intends.
    pytest.mark.xdist_group("browser"),
    needs_dex,
    pytest.mark.skipif(ASSETS is None, reason="AID_WEB_ASSETS is not set"),
]


@pytest.fixture(autouse=True)
async def clean_sessions(session_daemon: Paths) -> AsyncIterator[None]:
    """Each test deletes its sessions when done: their workers exit, so the next test starts near-empty
    instead of accumulating a worker per session until the guest falls over."""
    yield
    async with aid.connect(session_daemon) as client:
        # Children go with their parent, so only the sessions without one.
        for session in await client.sessions():
            if session.parent is None:
                await client.session(session.name).delete()


async def test_pwa_is_installable(page: Page, browser_name: str) -> None:
    if browser_name != "chromium":
        pytest.skip("installability is a chromium concept")
    assert await page.locator('link[rel="manifest"]').count() >= 1
    cdp = await page.context.new_cdp_session(page)
    # The CDP bridge is untyped; the verdict below is a plain JSON object.
    result = cast(dict[str, JsonValue], await cdp.send("Page.getInstallabilityErrors"))  # pyright: ignore[reportUnknownMemberType]
    assert result.get("installabilityErrors", []) == []


async def test_login_shows_the_user(page: Page) -> None:
    # A fresh tab selects nothing, whatever other tests seeded: the empty prompt is a placeholder paragraph,
    # not a chat row, because no session has spoken yet.
    with anyio.fail_after(TIMEOUT):
        await page.get_by_text("Select or create a session.").wait_for(timeout=TIMEOUT * 1000)
    assert (await page.get_by_test_id("user-email").text_content()) == ALLOWED


async def test_working_directory_starts_at_home_and_completes(page: Page, tmp_path: Path) -> None:
    (tmp_path / "project").mkdir()
    cwd = page.get_by_test_id("cwd-input")
    home = str(Path.home()).rstrip("/") + "/"
    with anyio.fail_after(TIMEOUT):
        while await cwd.input_value() != home:  # noqa: ASYNC110 -- the page fills it once aid web answers
            await anyio.sleep(0.05)
    await cwd.fill(f"{tmp_path}/pro")
    option = page.locator(f'#cwd-completions option[value="{tmp_path}/project/"]')
    await option.wait_for(state="attached", timeout=TIMEOUT * 1000)


async def test_the_create_form_sets_a_tool_ceiling(
    page: Page, session_daemon: Paths, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Lillecarl/aid#50: one regex per line in the form becomes the native spec's `tools`."""
    # The daemon lists agents in a subprocess that reads the plugins path when asked.
    monkeypatch.setenv(ENV_PLUGINS_PATH, str(AGENT_DIR))
    await page.reload()
    form = page.locator("form").filter(has_text="New session")
    await form.get_by_label("Name").fill("ceiled")
    await form.get_by_label("Kind").select_option("native")
    await page.get_by_test_id("cwd-input").fill(str(tmp_path))
    agent = page.get_by_test_id("agent-select")
    await agent.locator("option", has_text="shouter").wait_for(state="attached", timeout=TIMEOUT * 1000)
    await agent.select_option("shouter")
    await page.get_by_test_id("tool-ceiling").fill("ls\n  read|outline \n\n")
    await form.get_by_role("button", name="Create").click()
    spec_file = session_daemon.state_dir / "sessions" / "ceiled" / "spec.json"
    with anyio.fail_after(TIMEOUT):
        while not spec_file.exists():  # noqa: ASYNC110 -- the daemon writes it once the page's request lands
            await anyio.sleep(0.05)
    assert json.loads(spec_file.read_text())["tools"] == ["ls", "read|outline"]


async def test_slash_cd_moves_the_session(page: Page, session_daemon: Paths, tmp_path: Path) -> None:
    (tmp_path / "there").mkdir()
    await seed_acp_session(session_daemon, tmp_path, "mover")
    await open_session(page, "mover")
    await send_chat(page, "/cd there")
    await expect_chat_text(page, f"Working directory changed ({(tmp_path / 'there').resolve()})")


async def test_a_person_approves_the_agents_move(page: Page, session_daemon: Paths, tmp_path: Path) -> None:
    (tmp_path / "sub").mkdir()
    await seed_session(session_daemon, tmp_path, "asker", "agents:coder", PermissionMode.ALLOW)
    async with aid.connect(session_daemon) as client:
        await client.session("asker").run(
            json.dumps([["change_directory", {"path": "sub", "reason": "tests live there"}]])
        )
    await open_session(page, "asker")
    card = page.get_by_test_id("move-request")
    await card.get_by_text("tests live there").wait_for(timeout=TIMEOUT * 1000)
    await card.get_by_role("button", name="Approve").click()
    await card.wait_for(state="detached", timeout=TIMEOUT * 1000)
    await expect_chat_text(page, f"this session's working directory is now {(tmp_path / 'sub').resolve()}")


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
    assert await items.count() == 3
    rendered = [await items.nth(i).text_content() for i in range(3)]
    assert any("/help" in (t or "") and "List these commands" in (t or "") for t in rendered)
    assert any("/cd" in (t or "") for t in rendered)
    assert any("/compact" in (t or "") for t in rendered)
    await box.press("Escape")
    await menu.wait_for(state="hidden", timeout=TIMEOUT * 1000)
    assert await box.input_value() == "/"


async def test_slash_menu_completes_keyboard(page: Page, session_daemon: Paths, tmp_path: Path) -> None:
    await seed_session(session_daemon, tmp_path, "complete", "agents:echo")
    await open_session(page, "complete")
    box = page.get_by_test_id("chat-input")
    await box.fill("/co")
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
    await seed_session(session_daemon, tmp_path, "tooling", "agents:tool_caller", AID_PLUGINS_PATH=str(AGENT_DIR))
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


async def test_ask_user_answers_from_the_card(page: Page, session_daemon: Paths, tmp_path: Path) -> None:
    await seed_session(session_daemon, tmp_path, "asker", "agents:coder", PermissionMode.ASK)
    await open_session(page, "asker")
    await send_chat(page, '[["ask_user", {"question": "Which database?", "options": ["postgres", "sqlite"]}]]')
    box = page.get_by_test_id("permission-box").filter(has_text="Which database?")
    await box.wait_for(timeout=TIMEOUT * 1000)
    await box.get_by_test_id("permission-text").fill("it must embed")
    await box.get_by_role("button", name="sqlite").click()
    await expect_chat_text(page, 'picked "sqlite" and added: it must embed')
    await wait_for_idle(page)
    await box.get_by_text("sqlite: it must embed").wait_for(timeout=TIMEOUT * 1000)


async def test_ask_user_marks_recommended(page: Page, session_daemon: Paths, tmp_path: Path) -> None:
    await seed_session(session_daemon, tmp_path, "asker", "agents:coder", PermissionMode.ASK)
    await open_session(page, "asker")
    await send_chat(
        page,
        '[["ask_user", {"question": "Which database?", "options": ["postgres", "sqlite"], "recommended": "sqlite"}]]',
    )
    box = page.get_by_test_id("permission-box").filter(has_text="Which database?")
    await box.wait_for(timeout=TIMEOUT * 1000)
    await box.get_by_role("button", name="sqlite (recommended)").wait_for(timeout=TIMEOUT * 1000)
    await box.get_by_text("If nobody answers, sqlite is picked automatically.").wait_for(timeout=TIMEOUT * 1000)
    await box.get_by_role("button", name="sqlite (recommended)").click()
    await expect_chat_text(page, 'picked "sqlite"')
    await wait_for_idle(page)


async def test_tool_card_names_its_file(page: Page, session_daemon: Paths, tmp_path: Path) -> None:
    await anyio.Path(tmp_path / "notes.txt").write_text("remember this\n")
    await seed_session(session_daemon, tmp_path, "reader", "agents:coder", PermissionMode.ALLOW)
    await open_session(page, "reader")
    await send_chat(page, '[["read", {"path": "notes.txt"}]]')
    await expect_chat_text(page, "remember this")
    await wait_for_idle(page)
    box = page.get_by_test_id("tool-box").filter(has_text="notes.txt")
    await box.wait_for(timeout=TIMEOUT * 1000)
    await open_details(box.locator("summary"), box.get_by_test_id("tool-paths"))
    assert "notes.txt" in (await box.get_by_test_id("tool-paths").text_content() or "")


async def test_a_shell_command_is_highlighted(page: Page, session_daemon: Paths, tmp_path: Path) -> None:
    """Lillecarl/aid#39: code the chat shows gets aid web's spans, here bash in a tool card."""
    await seed_session(session_daemon, tmp_path, "painter", "agents:coder", PermissionMode.ALLOW)
    await open_session(page, "painter")
    await send_chat(page, json.dumps([["shell", {"command": "echo 'painted' | tr a-z A-Z"}]]))
    await expect_chat_text(page, "PAINTED")
    await wait_for_idle(page)
    box = page.get_by_test_id("tool-box").filter(has_text="echo")
    await open_details(box.locator("summary"), box.locator("code.hl"))
    await box.locator("code.hl span.s, code.hl span.s1, code.hl span.s2").first.wait_for(timeout=TIMEOUT * 1000)
    assert "echo 'painted' | tr a-z A-Z" in (await box.locator("code.hl").text_content() or "")


async def test_a_read_file_is_highlighted_by_its_path(page: Page, session_daemon: Paths, tmp_path: Path) -> None:
    """Lillecarl/aid#39: `read` output shows its line numbers apart, the text highlighted as the file's language."""
    await anyio.Path(tmp_path / "paint.py").write_text("def paint():\n    return 'red'\n")
    await seed_session(session_daemon, tmp_path, "painter", "agents:coder", PermissionMode.ALLOW)
    await open_session(page, "painter")
    await send_chat(page, json.dumps([["read", {"path": "paint.py", "limit": 1}]]))
    await expect_chat_text(page, "def paint")
    await wait_for_idle(page)
    box = page.get_by_test_id("tool-box").filter(has_text="paint.py")
    listing = box.get_by_test_id("tool-listing")
    await open_details(box.locator("summary"), listing)
    await listing.locator("code.hl span.k").first.wait_for(timeout=TIMEOUT * 1000)
    assert await listing.locator(".gutter").all_text_contents() == ["1"]
    assert (await listing.locator(".line").all_text_contents()) == ["def paint():\n"]
    assert "1 more lines" in (await listing.locator(".note").text_content() or "")


async def test_a_clipped_read_keeps_its_line_numbers(page: Page, session_daemon: Paths, tmp_path: Path) -> None:
    """The history clips a long read inside line 199's number: the part line goes, the rest stays numbered."""
    # 101 characters a numbered line: the 20 000-character clip falls 2 into line 199.
    lines = [f"v{n:03d} = {'1' * 86}" for n in range(1, 251)]
    await anyio.Path(tmp_path / "long.py").write_text("\n".join(lines) + "\n")
    await seed_session(session_daemon, tmp_path, "longread", "agents:coder", PermissionMode.ALLOW)
    await open_session(page, "longread")
    await send_chat(page, json.dumps([["read", {"path": "long.py"}]]))
    await wait_for_idle(page)
    box = page.get_by_test_id("tool-box").filter(has_text="long.py")
    listing = box.get_by_test_id("tool-listing")
    await open_details(box.locator("summary"), listing)
    gutters = await listing.locator(".gutter").all_text_contents()
    assert gutters[0] == "1" and gutters[-1] == "198" and len(gutters) == 198
    assert "more characters" in (await listing.locator(".note").text_content() or "")


async def test_rows_fold_one_at_a_time_and_all_at_once(page: Page, session_daemon: Paths, tmp_path: Path) -> None:
    """Lillecarl/aid#40: a text row folds to its first line; Fold all also shuts every box."""
    await anyio.Path(tmp_path / "notes.txt").write_text("remember this\n")
    await seed_session(session_daemon, tmp_path, "folder", "agents:coder", PermissionMode.ALLOW)
    await open_session(page, "folder")
    await send_chat(page, '[\n["read",\n{"path": "notes.txt"}]\n]')
    await expect_chat_text(page, "remember this")
    await wait_for_idle(page)
    box = page.get_by_test_id("tool-box").filter(has_text="notes.txt")
    await open_details(box.locator("summary"), box.get_by_test_id("tool-paths"))
    row = page.locator(".user").last
    tall = await row.bounding_box()
    await row.get_by_test_id("fold").click()
    await page.locator(".user.folded").wait_for(timeout=TIMEOUT * 1000)
    short = await row.bounding_box()
    assert tall is not None and short is not None and short["height"] < tall["height"]
    await row.get_by_test_id("fold").click()
    await page.locator(".user.folded").wait_for(state="detached", timeout=TIMEOUT * 1000)
    await page.get_by_test_id("fold-all").click()
    await page.locator(".user.folded").first.wait_for(timeout=TIMEOUT * 1000)
    assert await page.locator("details[open]").count() == 0
    await page.get_by_test_id("unfold-all").click()
    assert await page.locator(".folded").count() == 0


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


async def test_permissions_tab_sets_checked_tools(page: Page, session_daemon: Paths, tmp_path: Path) -> None:
    await seed_session(session_daemon, tmp_path, "gated", "agents:coder", PermissionMode.ASK)
    async with aid.connect(session_daemon) as client:
        await client.session("gated").start()
    await open_session(page, "gated")
    await page.get_by_role("tab", name="Permissions").click()
    coding = page.get_by_test_id("permission-group").filter(has_text="coding")
    await coding.locator("legend input").check()
    await coding.locator("legend input").uncheck()
    ls = page.get_by_test_id("permission-tool").filter(has=page.get_by_text("ls", exact=True))
    await ls.get_by_role("checkbox").check()
    await page.get_by_test_id("permission-actions").get_by_role("button", name="deny").click()
    await ls.get_by_text("deny", exact=True).wait_for(timeout=TIMEOUT * 1000)
    async with aid.connect(session_daemon) as client:
        modes = {t.name: (t.mode, t.set) for t in (await client.session("gated").status()).tools}
    assert modes["ls"] == (PermissionMode.DENY, True)
    assert modes["read"] == (PermissionMode.ALLOW, False)


async def test_permissions_tab_sets_the_supervisor(page: Page, session_daemon: Paths, tmp_path: Path) -> None:
    await seed_session(session_daemon, tmp_path, "boss", "agents:echo")
    await seed_session(session_daemon, tmp_path, "worker", "agents:echo")
    await open_session(page, "worker")
    await page.get_by_role("tab", name="Permissions").click()
    picker = page.get_by_test_id("supervisor")
    await picker.locator("option[value=boss]").wait_for(state="attached", timeout=TIMEOUT * 1000)
    await picker.select_option("boss")
    async with aid.connect(session_daemon) as client:
        with anyio.fail_after(TIMEOUT):
            while (await client.session("worker").status()).supervisor is None:  # noqa: ASYNC110 -- the page saves it
                await anyio.sleep(0.05)
    await picker.select_option("")
    async with aid.connect(session_daemon) as client:
        with anyio.fail_after(TIMEOUT):
            while (await client.session("worker").status()).supervisor is not None:  # noqa: ASYNC110 -- the same
                await anyio.sleep(0.05)


async def test_linked_sessions_are_tabs_under_the_chat(page: Page, session_daemon: Paths, tmp_path: Path) -> None:
    """Lillecarl/aid#41: a supervisor and its classifier leave the list and show as tabs under each session it
    watches; a tab switches the whole view to that session."""
    await seed_session(session_daemon, tmp_path, "boss", "agents:echo")
    await seed_session(session_daemon, tmp_path, "worker", "agents:echo")
    async with aid.connect(session_daemon) as client:
        spec = py_spec(tmp_path, "agents:echo", PermissionMode.DENY).model_copy(update={"parent": "boss"})
        await client.create("boss.classifier", spec)
        await client.session("worker").set_supervisor("boss")
    listing = page.get_by_test_id("session-list")
    await page.get_by_test_id("show-linked").filter(has_text="Show 2 linked sessions").wait_for(timeout=TIMEOUT * 1000)
    assert await listing.get_by_role("button").filter(has_text="boss").count() == 0
    await open_session(page, "worker")
    tabs = page.locator("[data-testid=linked-tab]:visible")
    await tabs.filter(has_text="classifier").wait_for(timeout=TIMEOUT * 1000)
    assert [await t.text_content() for t in await tabs.all()] == ["worker", "boss", "classifier"]
    await tabs.filter(has_text="boss").click()
    await page.get_by_role("heading", name="worker \u203a boss").wait_for(timeout=TIMEOUT * 1000)
    box = page.locator("[data-testid=chat-input]:visible")
    await box.fill("hello boss")
    await box.press("Enter")
    await page.locator("[data-testid=chat-log]:visible").get_by_text("echo hello boss").wait_for(timeout=TIMEOUT * 1000)
    await page.get_by_role("tab", name="Status").click()
    await page.get_by_role("tab", name="Chat").click()
    await page.locator("[data-testid=linked-tab]:visible").filter(has_text="worker").click()
    await page.get_by_role("heading", name="worker", exact=True).wait_for(timeout=TIMEOUT * 1000)
    assert await page.locator("[data-testid=chat-log]:visible").get_by_text("echo hello boss").count() == 0
    await page.get_by_test_id("show-linked").click()
    await listing.get_by_role("button").filter(has_text="boss.classifier").wait_for(timeout=TIMEOUT * 1000)


async def test_tags_label_and_filter_sessions(page: Page, session_daemon: Paths, tmp_path: Path) -> None:
    """Lillecarl/aid#42: tags are set under a session's heading, shown in the list, and filter it."""
    await seed_session(session_daemon, tmp_path, "web", "agents:echo")
    await seed_session(session_daemon, tmp_path, "api", "agents:echo")
    await open_session(page, "web")
    await page.get_by_test_id("tag-input").fill("team=front stage=review")
    await page.get_by_test_id("tag-input").press("Enter")
    listing = page.get_by_test_id("session-list")
    await listing.get_by_test_id("tag").filter(has_text="team=front").wait_for(timeout=TIMEOUT * 1000)
    await page.get_by_role("button", name="Remove tag stage").click()
    async with aid.connect(session_daemon) as client:
        with anyio.fail_after(TIMEOUT):
            while {s.name: s.tags for s in await client.sessions()}["web"] != {"team": "front"}:  # noqa: ASYNC110
                await anyio.sleep(0.05)
    await page.get_by_test_id("tag-filter").fill("team=front")
    await listing.get_by_role("button").filter(has_text="api").wait_for(state="detached", timeout=TIMEOUT * 1000)
    assert await listing.get_by_role("button").filter(has_text="web").count() == 1
    await page.get_by_test_id("tag-filter").fill("team=back")
    await listing.get_by_text("No session has these tags.").wait_for(timeout=TIMEOUT * 1000)


@needs_bwrap
async def test_permissions_tab_turns_the_sandbox_on(page: Page, session_daemon: Paths, tmp_path: Path) -> None:
    await seed_session(session_daemon, tmp_path, "boxed", "agents:coder", PermissionMode.ASK)
    await open_session(page, "boxed")
    await page.get_by_role("tab", name="Permissions").click()
    box = page.get_by_test_id("sandbox")
    await box.get_by_test_id("sandbox-on").check()
    await box.get_by_label("Network").uncheck()
    await box.get_by_test_id("sandbox-save").click()
    async with aid.connect(session_daemon) as client:
        with anyio.fail_after(TIMEOUT):
            while (await client.session("boxed").status()).sandbox is None:  # noqa: ASYNC110 -- the page saves it
                await anyio.sleep(0.05)
        status = await client.session("boxed").status()
    assert status.sandbox == Sandbox(network=False)


async def test_fork_usage_names_the_fork(page: Page, session_daemon: Paths, tmp_path: Path) -> None:
    """A fork's tokens ride its answer into the parent; the row names the fork, not a Claude subagent."""
    await seed_session(session_daemon, tmp_path, "forker", "agents:echo")
    await open_session(page, "forker")
    usage = Usage(input_tokens=1200, models=["local"], agent="fork forker.fork-abcdef")
    async with aid.connect(session_daemon) as client:
        await client.send_message("forker", "the answer", sender="forker.fork-abcdef", usage=usage)
    await expect_chat_text(page, "fork forker.fork-abcdef · local · 1.2k in")
    assert await page.get_by_test_id("chat-row").filter(has_text="subagent fork").count() == 0


async def test_a_priced_turn_shows_its_cost(page: Page, session_daemon: Paths, tmp_path: Path) -> None:
    """aid's own price for a turn (native): a fraction of a cent reads as such, not as 0.00."""
    await seed_session(session_daemon, tmp_path, "priced", "agents:echo")
    await open_session(page, "priced")
    usage = Usage(input_tokens=1200, models=["local"], cost=Cost(amount=0.0011324, currency="USD"))
    async with aid.connect(session_daemon) as client:
        await client.send_message("priced", "the answer", sender="priced.fork-abcdef", usage=usage)
    await expect_chat_text(page, "local · 1.2k in · 0.0011 USD")
    await wait_for_idle(page)
    await page.get_by_role("tab", name="Status").click()
    cost = page.get_by_test_id("status-cost")
    await cost.wait_for(timeout=TIMEOUT * 1000)
    assert await cost.text_content() == "0.0011 USD"


async def test_usertee_output_shows_live_until_the_turn_ends(page: Page, session_daemon: Paths, tmp_path: Path) -> None:
    """What a command pipes through `usertee` shows while the turn runs, though `tail` keeps only its last line
    from the agent, and goes when the turn ends: it is never recorded."""
    gate = tmp_path / "tee-gate"
    command = f"printf 'step 1\\nstep 2\\n' | usertee | tail -n 1; while [ ! -e {gate} ]; do sleep 0.1; done"
    await seed_session(session_daemon, tmp_path, "teeing", "agents:coder", PermissionMode.ALLOW)
    await open_session(page, "teeing")
    await send_chat(page, json.dumps([["shell", {"command": command}]]))
    block = page.get_by_test_id("live-output")
    await block.wait_for(timeout=TIMEOUT * 1000)
    assert "step 1\nstep 2" in (await block.locator("pre").text_content() or "")
    await anyio.Path(gate).write_text("go")
    await wait_for_idle(page)
    await block.wait_for(state="detached", timeout=TIMEOUT * 1000)


async def test_live_output_follows_its_end_until_scrolled_up(page: Page, session_daemon: Paths, tmp_path: Path) -> None:
    """The live block's own scroll area keeps to its last line while it is there, and stays where the person
    scrolled once they scroll up."""
    first, second = tmp_path / "first", tmp_path / "second"
    wait = "while [ ! -e {} ]; do sleep 0.1; done"
    command = (
        f"seq 1 300 | usertee >/dev/null; {wait.format(first)}; seq 301 400 | usertee >/dev/null; {wait.format(second)}"
    )
    await seed_session(session_daemon, tmp_path, "scroller", "agents:coder", PermissionMode.ALLOW)
    await open_session(page, "scroller")
    await send_chat(page, json.dumps([["shell", {"command": command}]]))
    pre = page.get_by_test_id("live-output").locator("pre")
    await expect_text(pre, "300")
    assert await pre.evaluate("p => p.scrollHeight - p.scrollTop - p.clientHeight") < 5
    await pre.evaluate("p => { p.scrollTop = 0; }")
    await anyio.Path(first).write_text("go")
    await expect_text(pre, "400")
    assert await pre.evaluate("p => p.scrollTop") == 0
    await anyio.Path(second).write_text("go")
    await wait_for_idle(page)


async def expect_text(locator: Locator, text: str) -> None:
    with anyio.fail_after(TIMEOUT):
        while True:
            if text in (await locator.text_content() or ""):
                return
            await anyio.sleep(0.05)


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
    # A thought stands on its own line, never beside tool cards.
    assert await page.locator("[data-testid=flow-row] details.thought").count() == 0
    await anyio.Path(gate).write_text("go")
    await wait_for_idle(page)
    await page.locator("details.thought:not([open])").wait_for(timeout=TIMEOUT * 1000)
    assert await page.locator("details.thought[open]").count() == 0
    assert await page.locator("details.thought").count() == 1
    assert "let me think" in (await page.locator("details.thought").text_content() or "")


async def test_an_open_wide_thought_scrolls_nothing_sideways(page: Page, session_daemon: Paths, tmp_path: Path) -> None:
    """Lillecarl/aid#36: a thought with a long code line or token stays inside the log at phone width."""
    await page.set_viewport_size({"width": 375, "height": 800})
    await seed_session(session_daemon, tmp_path, "wide", "agents:wide_thinker")
    await open_session(page, "wide")
    await send_chat(page, "think")
    await wait_for_idle(page)
    await page.locator("details.thought").wait_for(timeout=TIMEOUT * 1000)
    if not await page.locator("details.thought[open]").count():
        await open_details(page.locator("details.thought summary"), page.locator("details.thought[open]"))
    overflow = await page.evaluate(
        """() => {
            const log = document.querySelector("[data-testid=chat-log]");
            const page = document.scrollingElement;
            return [page.scrollWidth - page.clientWidth, log.scrollWidth - log.clientWidth];
        }"""
    )
    assert overflow == [0, 0]


async def test_a_wide_permission_card_scrolls_nothing_sideways(
    page: Page, session_daemon: Paths, tmp_path: Path
) -> None:
    """Lillecarl/aid#38: a request with a long line stays inside the log, and a person can fold it."""
    await page.set_viewport_size({"width": 375, "height": 800})
    await seed_session(session_daemon, tmp_path, "widecard", "agents:coder", PermissionMode.ASK)
    await open_session(page, "widecard")
    await send_chat(page, json.dumps([["shell", {"command": "echo " + "x" * 600}]]))
    box = page.get_by_test_id("permission-box")
    await box.locator("pre").wait_for(timeout=TIMEOUT * 1000)
    overflow = await page.evaluate(
        """() => {
            const log = document.querySelector("[data-testid=chat-log]");
            const page = document.scrollingElement;
            return [page.scrollWidth - page.clientWidth, log.scrollWidth - log.clientWidth];
        }"""
    )
    card = await box.bounding_box()
    log = await page.get_by_test_id("chat-log").bounding_box()
    assert card is not None and log is not None
    assert card["x"] + card["width"] <= log["x"] + log["width"]
    assert overflow == [0, 0]
    await box.locator("summary").click()
    await box.locator("pre").wait_for(state="hidden", timeout=TIMEOUT * 1000)
    await box.locator("summary").click()
    await box.get_by_role("button", name="Cancel").click()
    await wait_for_idle(page)


async def test_an_edit_request_shows_its_diff_as_lines(page: Page, session_daemon: Paths, tmp_path: Path) -> None:
    """apply_edits asks with `{"diff": ...}`: one JSON string, shown as diff lines rather than one escaped line."""
    await anyio.Path(tmp_path / "notes.txt").write_text("old line\n")
    await seed_session(session_daemon, tmp_path, "editor", "agents:coder", PermissionMode.ASK)
    await open_session(page, "editor")
    await send_chat(
        page, json.dumps([["edit", {"path": "notes.txt", "old": "old line", "new": "new line"}], ["apply_edits", {}]])
    )
    box = page.get_by_test_id("permission-box")
    await box.locator(".l.add", has_text="+new line").wait_for(timeout=TIMEOUT * 1000)
    assert await box.locator(".l.del", has_text="-old line").count() == 1
    await box.get_by_role("button", name="Cancel").click()
    await wait_for_idle(page)


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
    await open_details(page.locator("details.thought summary"), page.locator("details.thought[open]"))
