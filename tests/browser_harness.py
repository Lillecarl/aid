"""The web UI through a real browser: the few actions tests need, and the fixtures behind them.

`browser` launches once for the run, `logged_in_state` walks dex's form into a cookie file once, and the
function-scoped `page` mints an isolated tab from those cookies with the app already open: tests do their
own thing from the first line. The agent behind every session is a mock from tests/agents.py: `agents:echo`
answers with the prompt, `agents:summarizer` compacts to a fixed digest. No network, no cost, deterministic
text to assert on.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import anyio
import pytest

import aid
from tests.conftest import BROWSERS, acp_spec, py_spec, test_failed
from tests.web_harness import ALLOWED, TIMEOUT, Web

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from playwright.async_api import Browser, Page

    from aid.paths import Paths

try:
    from playwright.async_api import async_playwright, expect
except ImportError:
    async_playwright = None
    expect = None


async def seed_session(daemon: Paths, cwd: Path, name: str, target: str, **env: str) -> None:
    """A session on a mock agent, ready before the page loads: the UI under test never waits on a model."""
    async with aid.connect(daemon) as client:
        await client.create(name, py_spec(cwd, target, **cast("dict[str, Any]", env)))


async def seed_acp_session(daemon: Paths, cwd: Path, name: str, **env: str) -> None:
    """A session on the fake ACP agent: the only mock that reports cost."""
    async with aid.connect(daemon) as client:
        await client.create(name, acp_spec(cwd, **cast("dict[str, Any]", env)))


async def login_ui(page: Page, web: Web) -> None:
    """Through dex's own form, as a person: email and password, then the app shell."""
    with anyio.fail_after(TIMEOUT):
        await page.goto(f"{web.url}/")
        await page.locator('input[name="login"]').fill(ALLOWED)
        await page.locator('input[name="password"]').fill("password")
        await page.locator('input[name="password"]').press("Enter")
        # The session list mounting means the login round-tripped and the app rendered.
        await page.get_by_test_id("session-list").wait_for(timeout=TIMEOUT * 1000)


async def open_session(page: Page, name: str) -> None:
    """Click the session in the list; the chat box mounting means the session view loaded. The list fills
    over the events socket, so the button may arrive after the seed call returns."""
    with anyio.fail_after(TIMEOUT):
        button = page.get_by_test_id("session-list").get_by_role("button").filter(has_text=name)
        await button.wait_for(timeout=TIMEOUT * 1000)
        await button.click()
        await page.get_by_test_id("chat-input").wait_for(timeout=TIMEOUT * 1000)


async def send_chat(page: Page, text: str) -> None:
    """Fill the prompt box and press Enter, the path a keyboard person takes. Waits for the previous turn
    first: the page drops a send into a busy box, and dismisses the slash menu when the text opens one, so
    Enter sends what was filled. A test driving mid-turn behaviour uses the Stop button directly instead, and
    a test driving the menu presses its own keys."""
    if expect is None:
        pytest.skip("playwright is not installed")
    await wait_for_idle(page)
    box = page.get_by_test_id("chat-input")
    await box.fill(text)
    if re.fullmatch(r"/[A-Za-z]*", text) is not None:
        menu = page.get_by_test_id("slash-menu")
        await menu.wait_for(timeout=TIMEOUT * 1000)
        await box.press("Escape")
        await menu.wait_for(state="hidden")
    await box.press("Enter")


async def wait_for_idle(page: Page) -> None:
    """Wait for the running turn to end: the Send button enables when the page is no longer busy. Asserts
    through the button's state, never page JavaScript: the CSP forbids eval."""
    if expect is None:
        pytest.skip("playwright is not installed")
    await expect(page.get_by_test_id("chat-send")).to_be_enabled(timeout=TIMEOUT * 1000)


async def expect_chat_text(page: Page, text: str) -> None:
    """Wait for a chat row carrying the text: the agent answered, or the command printed."""
    with anyio.fail_after(TIMEOUT):
        await (
            page.get_by_test_id("chat-row")
            .filter(has_text=text)
            .first.wait_for(state="visible", timeout=TIMEOUT * 1000)
        )


#: How the fixture finds the browser the Nix shell provides in the requested browser's directory.
NEEDLES = {"chromium": "chromium_headless_shell", "firefox": "firefox"}
#: The sandbox forbids the namespaces chromium's sandbox needs, and its /dev/shm is tiny.
CHROMIUM_ARGS = ["--no-sandbox", "--disable-dev-shm-usage"]
#: The same namespaces forbid firefox's content sandbox: it dies on the first page without this.
FIREFOX_ENV = {"MOZ_DISABLE_CONTENT_SANDBOX": "1"}


def pytest_generate_tests(metafunc: pytest.Metafunc) -> None:
    """Run every browser test once per browser: `browser_name` carries the choice at session scope, so each
    browser gets its own browser, login, daemon and web stack."""
    if "browser_name" in metafunc.fixturenames:
        chosen = metafunc.config.getoption("--browser")
        names = list(BROWSERS) if chosen == "both" else [chosen]
        metafunc.parametrize("browser_name", names, indirect=True, scope="session")


@pytest.fixture(scope="session")
def browser_name(request: pytest.FixtureRequest) -> str:
    assert request.param in BROWSERS, f"--browser={request.param} is not one of {BROWSERS}"
    return request.param


def browsers_root() -> Path | None:
    value = os.environ.get("PLAYWRIGHT_BROWSERS_PATH")
    if value is None:
        return None
    root = Path(value)
    return root if root.is_dir() else None


@pytest.fixture(scope="session")
async def browser(browser_name: str) -> AsyncIterator[Browser]:
    """A real browser for the run: launched once, shared by every test through isolated contexts.
    Skips when the Nix shell provides no browser."""
    if async_playwright is None:
        pytest.skip("playwright is not installed")
    root = browsers_root()
    needle = NEEDLES[browser_name]
    if root is None or not list(root.glob(f"{needle}-*")):
        pytest.skip(f"no {browser_name} under PLAYWRIGHT_BROWSERS_PATH={root}")
    async with async_playwright() as p:
        launched = await {"chromium": p.chromium, "firefox": p.firefox}[browser_name].launch(
            headless=True,
            args=CHROMIUM_ARGS if browser_name == "chromium" else [],
            env={**os.environ, **FIREFOX_ENV} if browser_name == "firefox" else None,
        )
        yield launched
        await launched.close()


@pytest.fixture(scope="session")
async def logged_in_state(
    browser: Browser, session_web: Web, tmp_path_factory: pytest.TempPathFactory
) -> AsyncIterator[Path]:
    """The dex login, once for the run: a context walks the form, and its cookies reach every test as a
    file. A test that logs out poisons the rest, so none does."""
    state = tmp_path_factory.mktemp("login") / "storage-state.json"
    context = await browser.new_context()
    chat = await context.new_page()
    try:
        await login_ui(chat, session_web)
        await context.storage_state(path=state)
    finally:
        await context.close()
    yield state


@pytest.fixture
async def page(
    browser: Browser,
    logged_in_state: Path,
    session_web: Web,
    tmp_path: Path,
    request: pytest.FixtureRequest,
) -> AsyncIterator[Page]:
    """A logged-in tab, isolated per test: fresh cookies from the session's login, the app already open.
    Screenshots on failure, into the test's directory."""
    context = await browser.new_context(storage_state=str(logged_in_state), viewport={"width": 1280, "height": 900})
    chat = await context.new_page()
    with anyio.fail_after(TIMEOUT):
        await chat.goto(f"{session_web.url}/")
        # The session list mounting means the saved login still holds and the app rendered.
        await chat.get_by_test_id("session-list").wait_for(timeout=TIMEOUT * 1000)
    yield chat
    if test_failed(request):
        shot = tmp_path / "browser-failure.png"
        await chat.screenshot(path=str(shot))
        print(f"\nbrowser failure screenshot: {shot}")
    await context.close()
