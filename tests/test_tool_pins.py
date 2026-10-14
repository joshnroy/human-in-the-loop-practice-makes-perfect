"""The lint and type tools are pinned exactly, so the local gate and CI run the same
versions. An unpinned ruff let CI (0.17.0) flag code that the local 0.16.1 passed."""

import re
from pathlib import Path

PYPROJECT = Path(__file__).resolve().parents[1] / "pyproject.toml"


def test_ruff_mypy_and_import_linter_are_pinned_to_one_version() -> None:
    # Python 3.10 has no tomllib; the dev extra is a flat list of quoted specs.
    block = re.search(r"^dev = \[(.*?)^\]", PYPROJECT.read_text(), re.S | re.M)
    assert block is not None
    specs = re.findall(r'"([^"]+)"', block.group(1))
    pinned = {spec.split("==")[0]: spec for spec in specs if "==" in spec}
    for tool in ("ruff", "mypy", "import-linter"):
        assert tool in pinned, f"{tool} must be pinned with == in the dev extra"
