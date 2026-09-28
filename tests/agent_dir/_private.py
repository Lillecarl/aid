"""A leading underscore keeps a module out of the catalog; importing this one would fail the tests."""

raise AssertionError("the catalog imported a private module")
