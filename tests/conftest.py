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
    from collections.abc import AsyncIterator, Iterator

    from aid.speech import Recognizer

TESTS = Path(__file__).parent


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
