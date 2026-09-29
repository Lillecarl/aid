from __future__ import annotations

import json

from aid.web.zws import frames, pong, refusal


def test_a_multipart_message_sets_more_on_all_but_the_last() -> None:
    assert frames([b"topic", b"payload"]) == [b"\x01topic", b"\x00payload"]
    assert frames([b"one"]) == [b"\x00one"]


def test_pong_carries_the_ping_context() -> None:
    assert pong(b"\x04PING\x00\x0acontext") == b"\x02\x04PONGcontext"
    assert pong(b"\x04PING\x00\x0a" + b"x" * 20) == b"\x02\x04PONG" + b"x" * 16
    assert pong(b"\x05READY") is None
    assert pong(b"\x04PING") is None  # No TTL.


def test_a_refusal_names_the_request_when_it_can() -> None:
    assert json.loads(refusal(b'{"op": "hook", "id": "r1"}', "no"))["id"] == "r1"
    assert json.loads(refusal(b"not json", "no"))["id"] == ""
    assert json.loads(refusal(b'{"id": 5}', "no"))["id"] == ""
