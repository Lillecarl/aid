from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from aid.web.auth import OidcConfig, allowed

if TYPE_CHECKING:
    from pydantic import JsonValue

CONFIG = OidcConfig("https://idp", "aid", "secret", "https://aid", frozenset({"me@example.com"}))


@pytest.mark.parametrize(
    ("email", "verified", "expected"),
    [
        ("me@example.com", True, "me@example.com"),
        ("Me@Example.com", True, "Me@Example.com"),
        ("me@example.com", False, None),
        ("me@example.com", "true", None),
        ("other@example.com", True, None),
        (["me@example.com"], True, None),
    ],
)
def test_only_a_verified_listed_email_logs_in(email: JsonValue, verified: JsonValue, expected: str | None) -> None:
    assert allowed({"email": email, "email_verified": verified}, CONFIG) == expected
