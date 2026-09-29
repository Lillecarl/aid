from __future__ import annotations

import multiprocessing
from typing import TYPE_CHECKING

from aid.launcher import PRELOAD, ForkserverLauncher

if TYPE_CHECKING:
    from pytest import MonkeyPatch


def test_preload_keeps_worker_code_fresh() -> None:
    """Only the transport may be preloaded: aid and the agent frameworks import fresh in each worker, so a
    worker restart picks up new code and `python_path` can override them."""
    assert PRELOAD == ("zmq.asyncio",)
    assert not any(name == "aid" or name.startswith("aid.") for name in PRELOAD)
    assert not {"pydantic_ai", "acp"} & set(PRELOAD)


class _FakeContext:
    def __init__(self) -> None:
        self.preloads: list[list[str]] = []

    def set_forkserver_preload(self, modules: list[str]) -> None:
        self.preloads.append(modules)


def test_forkserver_launcher_passes_preload(monkeypatch: MonkeyPatch) -> None:
    # The forkserver imports its preload once, when it starts: in-process only the first launcher counts,
    # so this checks the handoff, not a running forkserver.
    context = _FakeContext()

    def fake_context(name: str) -> _FakeContext:
        assert name == "forkserver"
        return context

    monkeypatch.setattr(multiprocessing, "get_context", fake_context)
    ForkserverLauncher(["a", "b"])
    ForkserverLauncher()
    assert context.preloads == [["a", "b"], list(PRELOAD)]
