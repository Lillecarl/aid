from __future__ import annotations

import shutil
import sys
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING

import anyio
import pytest

from aid.daemon import Daemon
from aid.launcher import ForkserverLauncher
from aid.paths import Paths
from aid.spec import AcpSpec, PermissionMode, PydanticAISpec

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Iterator

TESTS = Path(__file__).parent


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


@pytest.fixture(scope="session")
def launcher() -> ForkserverLauncher:
    return ForkserverLauncher()


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


def acp_spec(cwd: Path, permission: PermissionMode = PermissionMode.DENY, **env: str) -> AcpSpec:
    return AcpSpec(
        cwd=str(cwd),
        command=[sys.executable, str(TESTS / "fake_acp_agent.py")],
        env=env,
        permission=permission,
    )


def py_spec(cwd: Path, target: str, **env: str) -> PydanticAISpec:
    return PydanticAISpec(cwd=str(cwd), target=target, python_path=[str(TESTS)], env=env)
