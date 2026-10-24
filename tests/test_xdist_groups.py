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
CI_WORKFLOW = TESTS.parent / ".github" / "workflows" / "ci.yml"


def test_every_ci_pytest_starts_one_worker_per_physical_core() -> None:
    """`-n auto`, not `-n logical`: the two extra hyperthreads measured as no gain at all.

    `-n auto` counts *physical* cores whenever psutil is importable, and it is here, so on
    a 4-vCPU `ubuntu-latest` runner (2 physical cores) it starts 2 workers: `test /
    tossing3d`'s junit times summed to 87.5 min against a 44 min wall clock. `-n logical`
    started 4, and every test ran about twice as slow: 170.6 min of junit time, 46 min of
    wall clock, and one shelf-pick test went from 38 s to 872 s.
    """
    commands = [
        line for line in CI_WORKFLOW.read_text(encoding="utf-8").splitlines() if "pytest -q" in line
    ]
    assert commands, "no pytest steps found"
    assert all(" -n auto " in line for line in commands), commands


def test_every_module_with_a_shared_fixture_is_an_xdist_group() -> None:
    missing = [
        str(path.relative_to(TESTS))
        for path in sorted(TESTS.rglob("test_*.py"))
        if SHARED_SCOPE.search(text := path.read_text(encoding="utf-8"))
        and "pytest.mark.xdist_group(" not in text
    ]
    assert not missing, f"add `pytestmark = pytest.mark.xdist_group(...)` to: {missing}"
