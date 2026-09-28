"""A module that fails to import: the catalog must report it and still list the rest."""

from __future__ import annotations

import no_such_module_for_aid_tests  # noqa: F401  # pyright: ignore[reportMissingImports, reportUnusedImport] -- missing on purpose
