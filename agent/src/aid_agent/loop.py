"""One turn of an agent: model requests, tool calls, continuations and retries, until the model answers.

`run` owns no state past the turn. The caller owns the history, the `Turn` (messages appended as they form, so a
cancel or a failure leaves them readable), the cancel scope, persistence and compaction.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Final

import anyio
import structlog
from anyio.lowlevel import checkpoint
from pydantic import TypeAdapter, ValidationError
from pydantic_core import to_jsonable_python

from aid_agent.agent import ModelRetry
from aid_agent.attachments import ToolAttachment, checked, kind_of
from aid_agent.events import AttachmentRef, Cost, TextDelta, ThoughtDelta, ToolCall, Usage, clip, to_json, tool_paths
from aid_agent.messages import (
    AttachmentPart,
    ModelMessage,
    ModelMessagesTypeAdapter,
    ModelRequest,
    ModelResponse,
    Part,
    RetryPromptPart,
    SystemPromptPart,
    TextPart,
    ThinkingPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)
from aid_agent.model import ModelInfo, RequestUsage, accumulate
from aid_models.errors import AidError
from aid_models.prices import CURRENCY
from aid_models.transport import StreamBroken

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable, Mapping, Sequence

    from pydantic import JsonValue

    from aid_agent.agent import Tool
    from aid_agent.attachments import AttachmentStore, Kind
    from aid_agent.model import Model
    from aid_models.prices import Price

log = structlog.get_logger(__name__)

THOUGHT_LIMIT: Final = 500
"""Characters of a past thinking part kept for replay: thought serves the turn, not the replay."""
CONTINUE: Final = (
    "Your reply above stopped at the output token limit, mid-sentence. Continue it from exactly where it stopped, "
    "as if it had never been interrupted. Do not restart or summarize it."
)
"""The request after an answer cut at the output limit. DeepSeek's prefix completion, which would continue it in
place, refuses any history holding a tool call (measured 2026-10-23). Not asked to repeat its last words: Claude
Haiku 5.5 refused that or restarted the answer in 2 of 3 tries, and continued this plainly in 3 of 3 (2026-10-24).
`join_continued` still overlaps a repeat a model gives on its own."""
OVERLAP: Final = 500
"""The longest overlap `join_continued` looks for."""
LEADERS: Final = " \t\n.…"
"""What a continuation's repeat may open with before the repeated words."""
HOLD: Final = 200
"""Characters of a continuation held from the stream until its repeat of the answer is known and left out."""
SEAM_PUNCTUATION: Final = ",.;:!?—"
"""Punctuation a space follows in prose, where `join_continued` restores one."""
STREAM_RETRIES: Final = 3
"""Broken streams in a row a turn sends again before it fails."""
STREAM_BACKOFF: Final = 5.0
"""Seconds before the first resend; each one after waits that much longer."""

type Emit = Callable[[TextDelta | ThoughtDelta | ToolCall], Awaitable[None]]
type Permit = Callable[[Tool, ToolCallPart, dict[str, JsonValue]], Awaitable[str | None]]
"""Asked before each call of a tool that does not ask by itself (`Tool.asks`): None runs it, a text refuses it and
goes to the model as the call's result. It may wait as long as a person takes."""
type Receive = Callable[[], Awaitable[str | None]]
"""Messages that arrived since the turn began, as one user prompt; asked before each request after the first."""
type History = Sequence[ModelMessage] | Callable[[], Sequence[ModelMessage]]
"""The conversation before the turn. A getter is read before each request: a compaction may replace it mid-turn."""


@dataclass
class Turn:
    """What a turn has done so far. The caller holds it, so it stays readable after a cancel or a failure."""

    messages: list[ModelMessage] = field(default_factory=list[ModelMessage])
    """The turn's messages, appended as they form. Unanswered tool calls are closed in place when the turn ends
    without an answer."""
    usage: Usage = field(default_factory=lambda: Usage(requests=0, models=[]))
    output: Any = None
    """The answer: a `str`, or the validated `output_type`."""
    stop_reason: str | None = None
    """`end_turn` or `max_tokens` once the model answered; None while running, and after a failure or a cancel."""
    in_flight: tuple[list[ToolCallPart], int] | None = None
    """The response's tool calls and the index of the one running."""


class UsageLimitExceeded(AidError):
    def __init__(self, limit: int) -> None:
        super().__init__("request_limit", f"turn reached its {limit} model-request budget")


def signed_thinking(model: Model) -> bool:
    """Whether past signed thinking replays whole: Anthropic rejects a signed block whose signature the wire
    dropped, and Gemini's signed thoughts replay the same way."""
    return model.model_name.startswith(("anthropic/", "google/"))


def prune_history(
    history: Sequence[ModelMessage], boundary: int, *, preserve_signed_thinking: bool = False
) -> list[ModelMessage]:
    """Trim past thinking in the send view, without changing stored history.

    The turn's own thinking and signed Anthropic blocks stay whole. A fixed boundary keeps sends monotonic.
    """
    pruned: list[ModelMessage] = []
    for index, message in enumerate(history):
        past = index < boundary
        parts: list[Part] = []
        changed = False
        for part in message.parts:
            if past and isinstance(part, ThinkingPart):
                if preserve_signed_thinking and part.signature:
                    parts.append(part)
                    continue
                thought = part.content[:THOUGHT_LIMIT] + "…" if len(part.content) > THOUGHT_LIMIT else part.content
                parts.append(part.model_copy(update={"content": thought, "signature": None}))
                changed = True
            else:
                parts.append(part)
        pruned.append(message.model_copy(update={"parts": parts}) if changed else message)
    return pruned


def estimate_tokens(messages: Sequence[ModelMessage]) -> int:
    """A rough token count: the serialized size over four."""
    return len(ModelMessagesTypeAdapter.dump_json(list(messages))) // 4


def context_tokens(history: Sequence[ModelMessage], boundary: int, *, preserve_signed_thinking: bool = False) -> int:
    """How many tokens a request over `history` would carry, in `estimate_tokens`' unit: rough but monotonic."""
    return estimate_tokens(prune_history(history, boundary, preserve_signed_thinking=preserve_signed_thinking))


def join_continued(answer: str, more: str) -> str:
    """An answer and its continuation, joined at their longest overlap when the continuation repeats the answer's
    last words, as models do unasked.
    Measured on deepseek-flash, a repeat can open with "...", change case, or trade a no-break space for a plain one:
    the match ignores all three. A repeat starts at a word, so the overlap must too: a single repeated word counts,
    and a chance match inside a word does not. A repeat can also rewrite the word the cut split ("carbon-d", then
    "carbon dioxide"), so that word may give way to the repeat's. After the repeat, a paragraph break before a
    lowercase word is a space. Without a repeat, a continuation drops the space at the cut; every such seam
    measured fell between words, so one goes back between a word or punctuation and a word."""
    start = len(more) - len(more.lstrip(LEADERS))
    rest = more[start:]
    cut_word = max(answer.rfind(" "), answer.rfind("\n")) + 1
    for kept in (answer, answer[:cut_word] if 0 < cut_word < len(answer) else None):
        if kept is not None and (size := _overlap(kept, rest)):
            after = rest[size:]
            if after[:1].isspace() and "\n" in after[: len(after) - len(after.lstrip())]:
                stripped = after.lstrip()
                after = f" {stripped}" if stripped[:1].islower() else after
            return kept + after
    if any(char in ".…" for char in more[:start]):
        more = rest
    if answer and more and more[0].isalnum() and (answer[-1].isalnum() or answer[-1] in SEAM_PUNCTUATION):
        return f"{answer} {more}"
    return answer + more


def unshown(answer: str, more: str) -> str:
    """What joining `more` adds to an `answer` already streamed. A rewritten cut word cannot be taken back, so the
    stream gets what follows the part both share."""
    joined = join_continued(answer, more)
    return joined[len(os.path.commonprefix([answer, joined])) :]  # noqa: RUF071 characters, not path components


def _overlap(answer: str, more: str) -> int:
    """The longest start of `more` that ends `answer`, loosely compared, and starts at a word there or with
    punctuation (Claude Haiku 5.5 repeated the comma a cut ended on); 0 if none."""
    tail, head = _loose(answer[-OVERLAP:]), _loose(more[:OVERLAP])
    for size in range(min(len(tail), len(head)), 0, -1):
        boundary = size == len(tail) or tail[-size - 1] == " " or not head[0].isalnum()
        if boundary and tail.endswith(head[:size]):
            return size
    return 0


def _loose(text: str) -> str:
    """Case and whitespace folded, character for character, so lengths match the original."""
    return "".join(" " if char.isspace() else lower if len(lower := char.lower()) == 1 else char for char in text)


def interrupted_messages(messages: list[ModelMessage]) -> list[ModelMessage]:
    """Close unanswered calls without replaying them. A failed or cancelled tool can already have effects.
    A response the provider never started is dropped: there is nothing in it to keep."""
    normalized: list[ModelMessage] = []
    for message in messages:
        if isinstance(message, ModelResponse) and not message.parts:
            continue
        calls = [part for part in message.parts if isinstance(part, ToolCallPart)]
        ids = [call.tool_call_id for call in calls]
        if calls and (any(not identity for identity in ids) or len(ids) != len(set(ids))):
            message = message.model_copy(
                update={
                    "parts": [
                        *(part for part in message.parts if not isinstance(part, ToolCallPart)),
                        TextPart("The provider returned invalid tool-call identities. These calls did not execute."),
                    ],
                    "invalid_tool_calls": [call.model_dump(mode="json") for call in calls],
                }
            )
        normalized.append(message)
    messages = normalized
    pending: dict[str, ToolCallPart] = {}
    for message in messages:
        for part in message.parts:
            if isinstance(part, ToolCallPart):
                pending[part.tool_call_id] = part
            elif isinstance(part, ToolReturnPart | RetryPromptPart) and part.tool_name is not None:
                pending.pop(part.tool_call_id, None)
    if not pending:
        return messages
    return [
        *messages,
        ModelRequest(
            parts=[
                ToolReturnPart(
                    tool_name=call.tool_name,
                    tool_call_id=call.tool_call_id,
                    content="The turn was interrupted before this tool returned. Its effects are unknown; "
                    "inspect the current state before retrying it.",
                    outcome="interrupted",
                )
                for call in pending.values()
            ]
        ),
    ]


def to_event(
    part: ToolCallPart | ToolReturnPart | RetryPromptPart,
    kind: str | None = None,
    attachments: Sequence[AttachmentPart] = (),
) -> ToolCall:
    if isinstance(part, ToolCallPart):
        try:
            args = part.args_as_dict()
        except ValueError:
            return ToolCall(
                tool_call_id=part.tool_call_id,
                tool=part.tool_name,
                title=part.tool_name,
                kind=kind,
                status="in_progress",
                input=part.args_as_json_str(),
            )
        return ToolCall(
            tool_call_id=part.tool_call_id,
            tool=part.tool_name,
            title=part.tool_name,
            kind=kind,
            status="in_progress",
            input=to_json(args),
            paths=tool_paths(args),
        )
    if isinstance(part, ToolReturnPart):
        return ToolCall(
            tool_call_id=part.tool_call_id,
            status="completed",
            output=clip(part.model_response_str()),
            attachments=[AttachmentRef(media_type=image.media_type, sha256=image.sha256) for image in attachments],
        )
    return ToolCall(tool_call_id=part.tool_call_id, status="failed", output=clip(part.model_response()))


def add_cost(total: Cost | None, price: Price, usage: RequestUsage, model: Model) -> Cost:
    # Asked of the object, never its class: a wrapper (aid's slots) hides the class.
    cached = usage.cache_read_tokens if getattr(model, "cache_reads_in_prompt", False) else 0
    amount = price.cost(
        max(0, usage.input_tokens - cached), usage.cache_read_tokens, usage.cache_write_tokens, usage.output_tokens
    )
    return Cost(amount=(total.amount if total is not None else 0.0) + amount, currency=CURRENCY)


def takes(model: Model, inputs: Sequence[str] | None, kind: Kind) -> bool:
    """False only where `inputs` (models.dev's `modalities.input`) lacks `kind`: an undescribed model is asked. A
    transport that reads PDFs wherever its models read images (`pdfs_as_images`) takes them on an image model."""
    if inputs is None or kind in inputs:
        return True
    return kind == "pdf" and "image" in inputs and getattr(model, "pdfs_as_images", False)


async def _no_emit(_event: TextDelta | ThoughtDelta | ToolCall) -> None:
    return None


def _counted(usage: Usage, request: RequestUsage, price: Price | None, model: Model) -> Usage:
    usage = usage.model_copy(
        update={
            "input_tokens": (usage.input_tokens or 0) + request.input_tokens,
            "output_tokens": (usage.output_tokens or 0) + request.output_tokens,
            "cache_read_tokens": (usage.cache_read_tokens or 0) + request.cache_read_tokens,
            "cache_write_tokens": (usage.cache_write_tokens or 0) + request.cache_write_tokens,
            "thought_tokens": usage.thought_tokens
            if request.reasoning_tokens is None
            else (usage.thought_tokens or 0) + request.reasoning_tokens,
        }
    )
    if price is not None:
        usage = usage.model_copy(update={"cost": add_cost(usage.cost, price, request, model)})
    return usage


async def run(
    model: Model,
    history: History,
    prompt: Sequence[Part],
    *,
    turn: Turn | None = None,
    instructions: str = "",
    tools: Sequence[Tool] = (),
    emit: Emit = _no_emit,
    permit: Permit | None = None,
    receive: Receive | None = None,
    settings: Mapping[str, JsonValue] | None = None,
    output_type: Any = str,
    output_retries: int = 3,
    max_continuations: int | None = 4,
    max_requests: int | None = None,
    attachments: AttachmentStore | None = None,
    inline: bool = False,
    price: Price | None = None,
    inputs: Sequence[str] | None = None,
    open_tail: int = 0,
) -> Turn:
    """Run one turn over `history` plus `prompt` (the parts of the turn's opening request) to the model's answer.

    `instructions` open the turn as a system prompt when the history is empty. `turn.messages` gets what the turn
    adds, the history excluded: append them to it for the next turn. `attachments` holds the bytes of the
    `AttachmentPart`s sent (`inline`: always as base64, never through a Files API) and of what tools return.
    `price` prices each request into `turn.usage.cost`; `inputs` lists what the model reads (models.dev's
    `modalities.input`), else every kind is tried. `open_tail`: messages at the history's end sent with their
    thinking whole, as the request that first sent them did.

    Raises `AidError`: `request_limit` past `max_requests` (None: no cap), `invalid_tool_call` for a response with
    missing or duplicate call ids, `model_unavailable` after `STREAM_RETRIES` broken streams in a row. A tool's
    `ModelRetry` goes to the model; any other exception from a tool ends the turn, as does an answer that fails
    `output_type` `output_retries` times in a row.
    """
    turn = Turn() if turn is None else turn
    messages = turn.messages
    by_name = {tool.name: tool for tool in tools}
    adapter = None if output_type is str else TypeAdapter[Any](output_type)
    signed = signed_thinking(model)
    info = ModelInfo(list(tools), output_type, dict(settings or {}), attachments, inline)

    def current() -> Sequence[ModelMessage]:
        return history() if callable(history) else history

    if instructions and not current():
        messages.append(ModelRequest(parts=[SystemPromptPart(instructions)]))
    messages.append(ModelRequest(parts=list(prompt)))
    retries = continued = broken = 0
    so_far: str | None = None
    first = True
    try:
        while True:
            await checkpoint()
            # Only before a request the turn sends anyway, so a message taken is always seen.
            if not first and receive is not None and (arrived := await receive()):
                messages.append(ModelRequest(parts=[UserPromptPart(arrived)]))
            first = False
            if max_requests is not None and (turn.usage.requests or 0) >= max_requests:
                raise UsageLimitExceeded(max_requests)
            turn.usage = turn.usage.model_copy(update={"requests": (turn.usage.requests or 0) + 1})
            response = ModelResponse(parts=[], model_name=model.model_name)
            messages.append(response)
            indexes: dict[int, int] = {}
            request_usage = RequestUsage()
            usage_observed = False
            truncated = False
            before = current()
            wire = prune_history(
                [*before, *messages[:-1]], max(0, len(before) - open_tail), preserve_signed_thinking=signed
            )
            held = None if so_far is None else ""
            try:
                stream = model.stream(wire, info)
                try:
                    async for delta in stream:
                        accumulate(response, indexes, delta)
                        if delta.model_name:
                            response.model_name = delta.model_name
                        truncated = truncated or delta.truncated
                        if delta.usage is not None:
                            usage_observed = True
                            for name in RequestUsage.__dataclass_fields__:
                                counts = [getattr(request_usage, name), getattr(delta.usage, name)]
                                known = [count for count in counts if count is not None]
                                setattr(request_usage, name, max(known) if known else None)
                        if isinstance(delta.part, TextPart) and delta.part.content:
                            if held is None:
                                await emit(TextDelta(text=delta.part.content))
                            elif len(held := held + delta.part.content) >= HOLD:
                                await emit(TextDelta(text=unshown(so_far or "", held)))
                                held = None
                        elif isinstance(delta.part, ThinkingPart) and delta.part.content:
                            await emit(ThoughtDelta(text=delta.part.content))
                        await checkpoint()
                    if held:
                        await emit(TextDelta(text=unshown(so_far or "", held)))
                finally:
                    with anyio.CancelScope(shield=True):
                        await stream.aclose()
                broken = 0
            except StreamBroken as error:
                if broken >= STREAM_RETRIES:
                    # A code of its own: a driver waits this out without counting it against the work.
                    raise AidError("model_unavailable", str(error)) from error
                broken += 1
                # The caller's transcript keeps what streamed of it; the conversation drops it.
                messages.pop()
                log.warning("stream_broken", error=str(error), retry=broken)
                await anyio.sleep(STREAM_BACKOFF * broken)
                continue
            finally:
                if usage_observed:
                    turn.usage = _counted(turn.usage, request_usage, price, model)
                resolved = response.model_name or model.model_name
                if resolved not in turn.usage.models:
                    turn.usage = turn.usage.model_copy(update={"models": [*turn.usage.models, resolved]})
            calls = [part for part in response.parts if isinstance(part, ToolCallPart)]
            if not calls:
                text = "".join(part.content for part in response.parts if isinstance(part, TextPart))
                answer = text if so_far is None else join_continued(so_far, text)
                # A request that adds nothing to the answer (all thinking, or only the repeat) would be cut or
                # stop the same way again.
                grew = len(answer) > len(so_far or "")
                if truncated and grew and (max_continuations is None or continued < max_continuations):
                    so_far, continued = answer, continued + 1
                    messages.append(ModelRequest(parts=[UserPromptPart(CONTINUE)]))
                    continue
                try:
                    output = answer if adapter is None else adapter.validate_json(answer)
                except ValidationError as error:
                    if retries >= output_retries:
                        raise
                    retries += 1
                    so_far, continued = None, 0
                    messages.append(ModelRequest(parts=[RetryPromptPart(str(error))]))
                    continue
                turn.output = output
                turn.stop_reason = "max_tokens" if truncated or not grew else "end_turn"
                return turn
            so_far, continued = None, 0
            ids = [call.tool_call_id for call in calls]
            if any(not call_id for call_id in ids) or len(ids) != len(set(ids)):
                raise AidError("invalid_tool_call", "response has missing or duplicate tool-call identities")
            for index, call in enumerate(calls):
                turn.in_flight = (calls, index)
                tool = by_name.get(call.tool_name)
                try:
                    await emit(to_event(call, tool.kind if tool is not None else None))
                    await checkpoint()
                    if tool is None:
                        raise ModelRetry(f"unknown tool {call.tool_name!r}")
                    try:
                        arguments = call.args_as_dict()
                    except ValueError as error:
                        if truncated:
                            raise ModelRetry(f"{error}: the response hit its output token limit") from error
                        raise ModelRetry(str(error)) from error
                    refusal = None if tool.asks or permit is None else await permit(tool, call, arguments)
                    result = await tool.call(arguments, call.tool_call_id) if refusal is None else refusal
                except ModelRetry as error:
                    part = RetryPromptPart(str(error), tool_name=call.tool_name, tool_call_id=call.tool_call_id)
                    returned: list[Part] = [part]
                else:
                    returned = await _returned(call, result, attachments, model, inputs)
                    part = returned[0]
                    assert isinstance(part, ToolReturnPart)
                messages.append(ModelRequest(parts=returned))
                await emit(to_event(part, attachments=[p for p in returned if isinstance(p, AttachmentPart)]))
    finally:
        turn.in_flight = None
        if turn.stop_reason is None:
            messages[:] = interrupted_messages(messages)


async def _returned(
    call: ToolCallPart, result: object, attachments: AttachmentStore | None, model: Model, inputs: Sequence[str] | None
) -> list[Part]:
    if not isinstance(result, ToolAttachment):
        return [ToolReturnPart(call.tool_name, to_jsonable_python(result), call.tool_call_id)]
    kind = kind_of(checked(result.data))
    if attachments is None or not takes(model, inputs, kind):
        note = f"{result.text}; this model takes no {'PDFs' if kind == 'pdf' else 'images'}"
        return [ToolReturnPart(call.tool_name, note, call.tool_call_id)]
    attachment = await attachments.add(result.data, call.tool_call_id)
    return [ToolReturnPart(call.tool_name, result.text, call.tool_call_id), attachment]
