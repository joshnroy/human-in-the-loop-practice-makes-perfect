"""The executed notebook, shared by every test that compares against Tom's engines."""

import hashlib
from pathlib import Path
from typing import Any

import matplotlib as mpl
import pytest

CELLS = Path(__file__).parent / "data" / "competence_models_notebook_cells.txt"
NOTEBOOK_SHA256 = "24a23f0f99ea63fcf5dfd056dc12adb4e3be7241f990fdaec3297d0c2a82eef1"
CELLS_SHA256 = "bc910346800d26a44e385fec886c8155220b538535db838a44d04aa1e6307e91"


@pytest.fixture(scope="session")
def notebook() -> dict[str, Any]:
    raw = CELLS.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == CELLS_SHA256
    assert NOTEBOOK_SHA256 in raw.decode()
    mpl.use("Agg")
    namespace: dict[str, Any] = {"__name__": "competence_models_notebook"}
    exec(compile(raw, str(CELLS), "exec"), namespace)  # noqa: S102
    return namespace
