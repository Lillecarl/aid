"""`aid_agent.loop.run` as a library, with no aid: pymux-ai's shape and the hooks a host passes."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import anyio
from pydantic import BaseModel

from aid_agent.agent import Tool
from aid_agent.attachments import AttachmentStore
from aid_agent.loop import Turn, run
from aid_agent.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    SystemPromptPart,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)
from aid_agent.model import ModelInfo, RequestUsage, StreamDelta, accumulate

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator, Sequence
    from pathlib import Path

    from pydantic import JsonValue

PNG = b"\x89PNG\r\n\x1a\n" + bytes(32)


@dataclass
class Scripted:
    """Answers each request with the next list of parts."""

    answers: list[Sequence[TextPart | ToolCallPart]]
    model_name: str = "scripted/test"
    seen: list[list[ModelMessage]] = field(default_factory=list[list[ModelMessage]])
    infos: list[ModelInfo] = field(default_factory=list[ModelInfo])

    async def stream(self, messages: list[ModelMessage], info: ModelInfo) -> AsyncGenerator[StreamDelta]:
        self.seen.append(messages)
        self.infos.append(info)
        yield StreamDelta(usage=RequestUsage(input_tokens=10, output_tokens=2))
        for index, part in enumerate(self.answers.pop(0)):
            yield StreamDelta(index, part)

    async def request(self, messages: list[ModelMessage], info: ModelInfo) -> ModelResponse:
        response = ModelResponse(parts=[])
        indexes: dict[int, int] = {}
        async for delta in self.stream(messages, info):
            accumulate(response, indexes, delta)
        return response


class Verdict(BaseModel):
    working: bool
    reason: str


async def test_a_judge_with_an_image_and_a_typed_answer(tmp_path: Path) -> None:
    store = AttachmentStore(tmp_path)
    model = Scripted([[TextPart('{"working": true, "reason": "tests run"}')]])
    image = store.put(PNG)
    turn = await run(
        model,
        [],
        [UserPromptPart("Is the agent working?"), image],
        instructions="Judge the pane.",
        output_type=Verdict,
        attachments=store,
        inline=True,
    )
    assert turn.output == Verdict(working=True, reason="tests run")
    assert turn.stop_reason == "end_turn"
    assert (turn.usage.requests, turn.usage.input_tokens, turn.usage.output_tokens) == (1, 10, 2)
    assert turn.usage.models == ["scripted/test"]
    system, opening, answer = turn.messages
    assert system.parts == [SystemPromptPart("Judge the pane.")]
    assert opening.parts[1] == image
    assert isinstance(answer, ModelResponse)
    assert model.infos[0].inline
    assert model.infos[0].output_type is Verdict
    # The next turn over this history opens with no second system prompt.
    again = Scripted([[TextPart('{"working": false, "reason": "idle"}')]])
    second = await run(
        again, turn.messages, [UserPromptPart("Now?")], instructions="Judge the pane.", output_type=Verdict
    )
    assert [type(m) for m in second.messages] == [ModelRequest, ModelResponse]
    assert again.seen[0][:3] == turn.messages


async def test_a_retry_follows_an_answer_that_fails_its_type() -> None:
    model = Scripted([[TextPart("yes")], [TextPart('{"working": true, "reason": "ok"}')]])
    turn = await run(model, [], [UserPromptPart("?")], output_type=Verdict)
    assert turn.output == Verdict(working=True, reason="ok")
    assert turn.usage.requests == 2


async def test_permit_refuses_and_receive_adds_a_message() -> None:
    ran: list[str] = []

    def shout(text: str) -> str:
        ran.append(text)
        return text.upper()

    calls = [ToolCallPart("shout", '{"text": "a"}', "c1"), ToolCallPart("shout", '{"text": "b"}', "c2")]
    model = Scripted([calls, [TextPart("done")]])
    asked: list[dict[str, JsonValue]] = []

    async def permit(tool: Tool, call: ToolCallPart, arguments: dict[str, JsonValue]) -> str | None:
        asked.append(arguments)
        return "not b" if arguments["text"] == "b" else None

    inbox = ["a message arrived"]

    async def receive() -> str | None:
        return inbox.pop() if inbox else None

    turn = await run(model, [], [UserPromptPart("go")], tools=[Tool(shout)], permit=permit, receive=receive)
    assert ran == ["a"]
    assert asked == [{"text": "a"}, {"text": "b"}]
    returns = [p for m in turn.messages for p in m.parts if isinstance(p, ToolReturnPart)]
    assert [(r.tool_call_id, r.content) for r in returns] == [("c1", "A"), ("c2", "not b")]
    assert turn.messages[-2].parts == [UserPromptPart("a message arrived")]
    assert turn.output == "done"


async def test_a_cancel_leaves_the_turn_readable_and_closed() -> None:
    started = anyio.Event()

    async def wait() -> str:
        started.set()
        await anyio.sleep_forever()
        return ""

    model = Scripted([[ToolCallPart("wait", "{}", "c1")]])
    turn = Turn()
    async with anyio.create_task_group() as group:

        async def go() -> None:
            await run(model, [], [UserPromptPart("go")], turn=turn, tools=[Tool(wait)])

        group.start_soon(go)
        await started.wait()
        assert turn.in_flight is not None
        group.cancel_scope.cancel()
    assert turn.stop_reason is None
    assert turn.in_flight is None
    assert turn.usage.requests == 1
    closing = turn.messages[-1].parts[0]
    assert isinstance(closing, ToolReturnPart)
    assert closing.outcome == "interrupted"


async def test_history_is_read_before_each_request() -> None:
    """A getter sees a history a compaction replaced in the middle of the turn."""
    history: list[ModelMessage] = [ModelRequest([UserPromptPart("old")]), ModelResponse([TextPart("old answer")])]

    def compact() -> str:
        history[:] = [ModelRequest([UserPromptPart("summary")])]
        return "compacted"

    model = Scripted([[ToolCallPart("compact", "{}", "c1")], [TextPart("ok")]])
    await run(model, lambda: history, [UserPromptPart("go")], tools=[Tool(compact)])
    assert model.seen[0][0].parts == [UserPromptPart("old")]
    assert model.seen[1][0].parts == [UserPromptPart("summary")]
