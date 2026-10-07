"""A leading underscore keeps a module out of the catalog; importing this one would fail the tests."""

from __future__ import annotations

raise AssertionError("the catalog imported a private module")
