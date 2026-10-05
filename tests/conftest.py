from __future__ import annotations

import multiprocessing.forkserver
import os
import shutil
import sys
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING

import anyio
import pytest

from aid.agents import ENV_AGENTS_PATH
from aid.daemon import Daemon
from aid.launcher import ForkserverLauncher
from aid.paths import Paths
from aid.spec import AcpSpec, ClaudeTtySpec, PermissionMode, PydanticAISpec
from aid.speech import load as load_speech
from aid.web.highlight import ENV_GRAMMARS, Grammars

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Generator, Iterator
    from typing import Any

    from playwright.async_api import Browser, Page

    from aid.speech import Recognizer

TESTS = Path(__file__).parent

# The dex/app stack both the HTTP tests and the browser tests boot; its fixtures come from there.
pytest_plugins = ["tests.web_harness"]

try:
    from playwright.async_api import async_playwright
except ImportError:
    async_playwright = None


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--browser",
        action="store",
        default="chromium",
        choices=("chromium", "firefox"),
        help="which real browser tests/test_browser.py drives (the sandbox runs chromium)",
    )


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item: pytest.Item, call: pytest.CallInfo[Any]) -> Generator[Any, Any, Any]:
    """Keep each phase's report on the node: the browser page fixture screenshots on failure."""
    outcome = yield
    setattr(item, f"rep_{call.when}", outcome.get_result())
    return outcome


def test_failed(request: pytest.FixtureRequest) -> bool:
    """Whether the test's call phase failed: the hook above records it; the browser fixture asks."""
    node = getattr(request, "node", None)
    report: Any = getattr(node, "rep_call", None)
    return report is not None and bool(report.failed)


#: What `--browser` accepts: full chromium is never needed, its headless shell runs the suite.
BROWSERS = ("chromium", "firefox")
#: How the fixture finds the browser the Nix shell provides in the requested browser's directory.
NEEDLES = {"chromium": "chromium_headless_shell", "firefox": "firefox"}
#: The sandbox forbids the namespaces chromium's sandbox needs, and its /dev/shm is tiny.
CHROMIUM_ARGS = ["--no-sandbox", "--disable-dev-shm-usage"]


@pytest.fixture(scope="session")
def browser_name(request: pytest.FixtureRequest) -> str:
    name = request.config.getoption("--browser")
    assert name in BROWSERS, f"--browser={name} is not one of {BROWSERS}"
    return name


def browsers_root() -> Path | None:
    value = os.environ.get("PLAYWRIGHT_BROWSERS_PATH")
    if value is None:
        return None
    root = Path(value)
    return root if root.is_dir() else None


@pytest.fixture
async def browser(browser_name: str) -> AsyncIterator[Browser]:
    """A real browser for the test: function-scoped, because async fixtures run on the function-scoped
    loop anyio provides. Skips when the Nix shell provides no browser."""
    if async_playwright is None:
        pytest.skip("playwright is not installed")
    root = browsers_root()
    needle = NEEDLES[browser_name]
    if root is None or not list(root.glob(f"{needle}-*")):
        pytest.skip(f"no {browser_name} under PLAYWRIGHT_BROWSERS_PATH={root}")
    async with async_playwright() as p:
        launched = await {"chromium": p.chromium, "firefox": p.firefox}[browser_name].launch(
            headless=True, args=CHROMIUM_ARGS if browser_name == "chromium" else []
        )
        yield launched
        await launched.close()


@pytest.fixture
async def page(browser: Browser, tmp_path: Path, request: pytest.FixtureRequest) -> AsyncIterator[Page]:
    """A fresh browser tab: screenshots on failure, into the test's directory."""
    context = await browser.new_context(viewport={"width": 1280, "height": 900})
    chat = await context.new_page()
    yield chat
    if test_failed(request):
        shot = tmp_path / "browser-failure.png"
        await chat.screenshot(path=str(shot))
        print(f"\nbrowser failure screenshot: {shot}")
    await context.close()


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


@pytest.fixture(scope="session", autouse=True)
def empty_agents_path(tmp_path_factory: pytest.TempPathFactory) -> Iterator[None]:
    """Keep the agents and tools of whoever runs the tests out of the sessions. Set before the forkserver starts,
    so every worker inherits it."""
    previous = os.environ.get(ENV_AGENTS_PATH)
    os.environ[ENV_AGENTS_PATH] = str(tmp_path_factory.mktemp("no-agents"))
    yield
    if previous is None:
        del os.environ[ENV_AGENTS_PATH]
    else:
        os.environ[ENV_AGENTS_PATH] = previous


@pytest.fixture(scope="session")
def launcher(empty_agents_path: None) -> ForkserverLauncher:
    launcher = ForkserverLauncher()
    # Workers inherit the forkserver's environment, frozen when it starts: start it before a test monkeypatches.
    multiprocessing.forkserver.ensure_running()
    return launcher


@pytest.fixture
def paths(tmp_path: Path) -> Iterator[Paths]:
    # ipc:// paths must fit in sun_path (108 bytes); pytest's tmp_path can be longer.
    runtime = Path(tempfile.mkdtemp(prefix="aid-"))
    yield Paths(runtime_dir=runtime, state_dir=tmp_path / "state")
    shutil.rmtree(runtime, ignore_errors=True)


@pytest.fixture
async def daemon(paths: Paths, launcher: ForkserverLauncher) -> AsyncIterator[Paths]:
    async with anyio.create_task_group() as tg:
        await tg.start(Daemon(paths, launcher).serve)
        yield paths
        tg.cancel_scope.cancel()


def acp_spec(
    cwd: Path, permission: PermissionMode = PermissionMode.DENY, *, permission_timeout: float = 1800, **env: str
) -> AcpSpec:
    return AcpSpec(
        cwd=str(cwd),
        command=[sys.executable, str(TESTS / "fake_acp_agent.py")],
        env=env,
        permission=permission,
        permission_timeout=permission_timeout,
    )


def py_spec(cwd: Path, target: str, permission: PermissionMode = PermissionMode.DENY, **env: str) -> PydanticAISpec:
    return PydanticAISpec(cwd=str(cwd), target=target, python_path=[str(TESTS)], env=env, permission=permission)


needs_pymux = pytest.mark.skipif(shutil.which("pymux") is None, reason="pymux is not on PATH")


@pytest.fixture
async def pymux_socket() -> AsyncIterator[str]:
    directory = Path(tempfile.mkdtemp(prefix="aid-pymux-"))
    socket = str(directory / "pymux.sock")
    yield socket
    await anyio.run_process(["pymux", "-S", socket, "kill-server"], check=False)
    shutil.rmtree(directory, ignore_errors=True)


def fake_spec(cwd: Path, socket: str, env: dict[str, str] | None = None, *, trust_cwd: bool = False) -> ClaudeTtySpec:
    """Interactive Claude as `fake_claude.py` plays it, in the pymux server on `socket`."""
    return ClaudeTtySpec(
        cwd=str(cwd),
        command=[sys.executable, str(TESTS / "fake_claude.py")],
        env={"CLAUDE_CONFIG_DIR": str(cwd / "claude-config"), "FAKE_CLAUDE_CHANNELS": "1", **(env or {})},
        pymux_socket=socket,
        trust_cwd=trust_cwd,
    )


SPEECH_MODEL = Path(os.environ["AID_TEST_SPEECH_MODEL"]) if os.environ.get("AID_TEST_SPEECH_MODEL") else None


@pytest.fixture(scope="session")
def speech_recognizer() -> Recognizer | None:
    """AID_TEST_SPEECH_MODEL, loaded once: loading takes a second or two. The dev shell and the Nix tests set it."""
    return load_speech(SPEECH_MODEL) if SPEECH_MODEL is not None else None


GRAMMARS = Path(os.environ[ENV_GRAMMARS]) if os.environ.get(ENV_GRAMMARS) else None
needs_grammars = pytest.mark.skipif(GRAMMARS is None, reason=f"{ENV_GRAMMARS} is not set")


@pytest.fixture(scope="session")
def grammars() -> Grammars | None:
    """AID_TREE_SITTER_GRAMMARS, which the dev shell and the Nix tests set."""
    return Grammars(GRAMMARS) if GRAMMARS is not None else None
