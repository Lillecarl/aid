"""Commands as values: build one with `cmd`, run it by awaiting it."""

from __future__ import annotations

import math
import os
import shlex
import signal
import subprocess
import time
from contextlib import AsyncExitStack, asynccontextmanager
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any

import anyio
import anyio.to_thread

from pyrun.result import Failed, Output, Result, Running, TimedOut
from pyrun.scope import Scope, current

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator, AsyncIterator, Awaitable, Callable, Generator, Mapping, Sequence

    from anyio.abc import ByteSendStream
    from anyio.streams.memory import MemoryObjectReceiveStream

    from pyrun.scope import Launched

type Arg = str | os.PathLike[str]


@dataclass(frozen=True)
class Command:
    argv: tuple[str, ...]
    cwd: str | None = None
    """Relative to the scope's cwd, or absolute."""
    env: Mapping[str, str | None] = field(default_factory=dict[str, "str | None"])
    """Merged onto the scope's environment; None removes a variable."""
    input: bytes | None = None
    timeout: float | None = None
    check: bool = True
    """A non-zero exit raises Failed. False makes it a Result like any other."""

    def __str__(self) -> str:
        return shlex.join(self.argv)

    def __await__(self) -> Generator[Any, None, Result]:
        return self.run().__await__()

    def __or__(self, other: Command | Pipeline) -> Pipeline:
        return Pipeline((self,)) | other

    def with_(self, **changes: Any) -> Command:
        return replace(self, **changes)

    async def run(self) -> Result:
        return await _in_scope(lambda scope: _run(self, scope))

    async def lines(self) -> AsyncIterator[str]:
        """stdout line by line as it comes. At the end, a failure raises as `run` would. Leaving early: use
        `contextlib.aclosing`, or the process is killed only when the iterator is collected."""
        async with self.start() as process:
            async for line in process:
                yield line
            result = await process.wait()
        if self.check and not result.ok:
            raise Failed(result)

    @asynccontextmanager
    async def start(self) -> AsyncGenerator[Process]:
        """The process while it runs: write to `stdin`, read lines from it, `wait()` for its Result. Leaving the
        block kills it if it still runs."""
        scope = current()
        async with AsyncExitStack() as stack:
            if scope is None:
                scope = await stack.enter_async_context(Scope())
            await scope.check(self)
            send, receive = anyio.create_memory_object_stream[bytes](math.inf)
            launched = await stack.enter_async_context(
                scope.launch(self, stdin=subprocess.PIPE, stdout=subprocess.PIPE, forward=send)
            )
            process = Process(launched, receive)
            if self.input is not None:
                await process.write(self.input)
                await process.close_stdin()
            yield process


def cmd(
    *argv: Arg,
    cwd: Arg | None = None,
    env: Mapping[str, str | None] | None = None,
    input: str | bytes | None = None,
    timeout: float | None = None,
    check: bool = True,
) -> Command:
    if not argv:
        raise ValueError("a command needs at least a program")
    return Command(
        argv=tuple(os.fspath(a) for a in argv),
        cwd=os.fspath(cwd) if cwd is not None else None,
        env=dict(env or {}),
        input=input.encode() if isinstance(input, str) else input,
        timeout=timeout,
        check=check,
    )


def sh(script: str, **options: Any) -> Command:
    """A shell script, the one way pyrun hands text to a shell."""
    return cmd("sh", "-c", script, **options)


async def run(*argv: Arg, **options: Any) -> Result:
    return await cmd(*argv, **options).run()


async def all(*commands: Command | Pipeline) -> list[Result]:
    """Run together. The first failure kills the others and raises."""
    results: list[Result | None] = [None] * len(commands)

    async def one(i: int, command: Command | Pipeline) -> None:
        results[i] = await command.run()

    try:
        async with anyio.create_task_group() as tg:
            for i, command in enumerate(commands):
                tg.start_soon(one, i, command)
    except* Exception as group:
        raise group.exceptions[0] from None
    return [r for r in results if r is not None]


@dataclass(frozen=True)
class Pipeline:
    """stdout of each stage into stdin of the next. It fails if any stage fails, except an earlier stage killed by
    SIGPIPE: that is the next stage having read all it wanted (`rg ... | head`)."""

    stages: tuple[Command, ...]

    def __str__(self) -> str:
        return " | ".join(str(s) for s in self.stages)

    def __or__(self, other: Command | Pipeline) -> Pipeline:
        return Pipeline(self.stages + (other.stages if isinstance(other, Pipeline) else (other,)))

    def __await__(self) -> Generator[Any, None, Result]:
        return self.run().__await__()

    async def run(self) -> Result:
        return await _in_scope(lambda scope: _run_pipeline(self.stages, scope))


class Process:
    """A running process, from `Command.start`."""

    def __init__(self, launched: Launched, stdout: MemoryObjectReceiveStream[bytes]) -> None:
        self._launched = launched
        self._stdout = stdout
        self._buffer = b""

    @property
    def pid(self) -> int:
        return self._launched.process.pid

    @property
    def id(self) -> str:
        return self._launched.id

    @property
    def _stdin(self) -> ByteSendStream:
        if self._launched.process.stdin is None:
            raise RuntimeError("the process has no stdin")
        return self._launched.process.stdin

    async def write(self, data: str | bytes) -> None:
        await self._stdin.send(data.encode() if isinstance(data, str) else data)

    async def close_stdin(self) -> None:
        await self._stdin.aclose()

    async def readline(self) -> str | None:
        """The next line of stdout, without its newline; None once stdout has ended."""
        while b"\n" not in self._buffer:
            try:
                self._buffer += await self._stdout.receive()
            except anyio.EndOfStream:
                if not self._buffer:
                    return None
                line, self._buffer = self._buffer, b""
                return line.decode(errors="replace")
        line, self._buffer = self._buffer.split(b"\n", 1)
        return line.decode(errors="replace")

    def __aiter__(self) -> AsyncIterator[str]:
        return self._lines()

    async def _lines(self) -> AsyncIterator[str]:
        while (line := await self.readline()) is not None:
            yield line

    async def wait(self) -> Result:
        """Its Result, once it exits. The Result's output is what was written so far."""
        await self._launched.process.wait()
        return _result(self._launched)


async def _in_scope[T](work: Callable[[Scope], Awaitable[T]]) -> T:
    scope = current()
    if scope is not None:
        return await work(scope)
    async with Scope() as scope:
        return await work(scope)


def _output(launched: Launched, which: str) -> Output:
    tee = launched.stdout if which == "stdout" else launched.stderr
    return tee.output() if tee is not None else Output(path=launched.record / which, head=b"", size=0)


def _result(launched: Launched) -> Result:
    code = launched.process.returncode
    ended = launched.ended if launched.ended is not None else time.monotonic()
    return Result(
        id=launched.id,
        command=launched.command,
        code=code if code is not None and code >= 0 else None,
        signal=-code if code is not None and code < 0 else None,
        stdout=_output(launched, "stdout"),
        stderr=_output(launched, "stderr"),
        duration=ended - launched.started,
        record=launched.record,
    )


async def _run(command: Command, scope: Scope) -> Result:
    result, timed_out = await _run_stages((command,), scope)
    [only] = result
    if timed_out is not None:
        raise TimedOut(str(command), command.timeout or math.inf, timed_out)
    if command.check and not only.ok:
        raise Failed(only)
    return only


async def _run_pipeline(stages: Sequence[Command], scope: Scope) -> Result:
    results, timed_out = await _run_stages(stages, scope)
    if timed_out is not None:
        timeouts = [s.timeout for s in stages if s.timeout is not None]
        raise TimedOut(" | ".join(str(s) for s in stages), min(timeouts), timed_out)
    for stage, result in zip(stages, results, strict=True):
        piped_away = result is not results[-1] and result.signal == signal.SIGPIPE
        if stage.check and not result.ok and not piped_away:
            raise Failed(result)
    return results[-1]


async def _run_stages(stages: Sequence[Command], scope: Scope) -> tuple[list[Result], list[Running] | None]:
    """Start the stages connected by pipes, wait for all of them under the smallest timeout they name."""
    for stage in stages:
        await scope.check(stage)
    timeouts = [s.timeout for s in stages if s.timeout is not None]
    launched: list[Launched] = []
    timed_out: list[Running] | None = None
    async with AsyncExitStack() as stack:
        upstream: int | None = None
        for i, stage in enumerate(stages):
            last = i == len(stages) - 1
            if upstream is not None:
                stdin = upstream
            else:
                stdin = subprocess.PIPE if stage.input is not None else subprocess.DEVNULL
            read, write = os.pipe() if not last else (None, None)
            stdout = write if write is not None else subprocess.PIPE
            launched.append(await stack.enter_async_context(scope.launch(stage, stdin=stdin, stdout=stdout)))
            # The children hold their own copies; ours would keep the pipe open past their end.
            if upstream is not None:
                os.close(upstream)
            if write is not None:
                os.close(write)
            upstream = read
        first = launched[0].process
        with anyio.move_on_after(min(timeouts) if timeouts else math.inf) as timer:
            if stages[0].input is not None and first.stdin is not None:
                await first.stdin.send(stages[0].input)
                await first.stdin.aclose()
            for each in launched:
                await each.process.wait()
        if timer.cancelled_caught:
            timed_out = []
            for each in launched:
                timed_out += await anyio.to_thread.run_sync(each.running)
    return [_result(each) for each in launched], timed_out
