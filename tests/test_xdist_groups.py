"""Every test module that shares an expensive fixture keeps it on one xdist worker.

CI runs `pytest -n auto --dist loadgroup`. Under xdist's default distribution a
module-scoped fixture is not shared: each worker that receives one of the module's tests
builds it again. For `test_sampler_draws.py` that measured as one ~70 s CLI run repeated
on three workers. `--dist loadgroup` sends a module's tests to a single worker only if the
module carries `pytest.mark.xdist_group`, so a module that adds such a fixture without the
mark silently pays the cost again on every worker it lands on.
"""

import re
from pathlib import Path

TESTS = Path(__file__).parent
SHARED_SCOPE = re.compile(r"""scope=["'](module|class|package)["']""")


def test_every_module_with_a_shared_fixture_is_an_xdist_group() -> None:
    missing = [
        str(path.relative_to(TESTS))
        for path in sorted(TESTS.rglob("test_*.py"))
        if SHARED_SCOPE.search(text := path.read_text(encoding="utf-8"))
        and "pytest.mark.xdist_group(" not in text
    ]
    assert not missing, f"add `pytestmark = pytest.mark.xdist_group(...)` to: {missing}"
