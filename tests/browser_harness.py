"""The web UI through a real browser: the few actions tests need.

The `browser` and `page` fixtures live in tests/conftest.py next to the daemon one; this module holds what a
test does with the page. The agent behind every session is a mock from tests/agents.py: `agents:echo`
answers with the prompt, `agents:summarizer` compacts to a fixed digest. No network, no cost, deterministic
text to assert on.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import anyio

import aid
from tests.conftest import py_spec
from tests.web_harness import ALLOWED, TIMEOUT, Web

if TYPE_CHECKING:
    from pathlib import Path

    from playwright.async_api import Page

    from aid.paths import Paths


async def seed_session(daemon: Paths, cwd: Path, name: str, target: str) -> None:
    """A session on a mock agent, ready before the page loads: the UI under test never waits on a model."""
    async with aid.connect(daemon) as client:
        await client.create(name, py_spec(cwd, target))


async def login_ui(page: Page, web: Web) -> None:
    """Through dex's own form, as a person: email and password, then the app shell."""
    with anyio.fail_after(TIMEOUT):
        await page.goto(f"{web.url}/")
        await page.locator('input[name="login"]').fill(ALLOWED)
        await page.locator('input[name="password"]').fill("password")
        await page.locator('input[name="password"]').press("Enter")
        # The session list mounting means the login round-tripped and the app rendered.
        await page.get_by_test_id("session-list").wait_for()


async def open_session(page: Page, name: str) -> None:
    """Click the session in the list; the chat box mounting means the session view loaded."""
    with anyio.fail_after(TIMEOUT):
        await page.get_by_test_id("session-list").get_by_role("button").filter(has_text=name).click()
        await page.get_by_test_id("chat-input").wait_for()


async def send_chat(page: Page, text: str) -> None:
    """Fill the prompt box and press Enter, the path a keyboard person takes."""
    box = page.get_by_test_id("chat-input")
    await box.fill(text)
    await box.press("Enter")


async def expect_chat_text(page: Page, text: str) -> None:
    """Wait for a chat row carrying the text: the agent answered, or the command printed."""
    with anyio.fail_after(TIMEOUT):
        await page.get_by_test_id("chat-row").filter(has_text=text).first.wait_for(state="visible")
