from __future__ import annotations

from typing import TYPE_CHECKING

from pydantic import BaseModel, ConfigDict, Field

from aid.protocol import (
    AidError,
    CallPlugin,
    CompactSession,
    GetHistory,
    GetPane,
    GetScreen,
    GetStatus,
    GetSummary,
    Hook,
    ListAgents,
    ListSessions,
    ReceiveMessages,
    RequestDirectory,
    SendMessage,
)

if TYPE_CHECKING:
    from aid.protocol import Request

ENV_AGENT_CONNECTION = "AID_AGENT_CONNECTION"
AGENT_ID_PREFIX = "agent:"


class AgentConnection(BaseModel):
    """Launch-scoped agent control credentials. Keep these out of specs, history, and command arguments."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    endpoint: str = Field(min_length=1)
    server_key: str = Field(min_length=40, max_length=40)
    public_key: str = Field(min_length=40, max_length=40)
    secret_key: str = Field(min_length=40, max_length=40, repr=False)
    trust_pem: str = ""


def authorize(request: Request, session: str) -> Request:
    """Bind an agent request to its authenticated session; administration belongs to the local control socket."""
    match request:
        case ListSessions() | ListAgents() | GetStatus() | GetHistory() | GetSummary() | GetScreen() | GetPane():
            return request
        case SendMessage():
            return request.model_copy(update={"sender": session, "usage": None})
        case CallPlugin():
            return request.model_copy(update={"caller": session})
        case CompactSession() | ReceiveMessages() | Hook() | RequestDirectory() if request.session == session:
            return request
        case _:
            raise AidError("forbidden", f"agent {session!r} may not send {request.op!r} with this authority")
