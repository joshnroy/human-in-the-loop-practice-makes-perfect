"""Pull requests never save the pip cache, so they cannot evict the MimicLabs one.

Each PR used to save its own ~3.5 GB `setup-python` pip cache. A handful of those
overflowed the repository's cache quota, GitHub evicted the 1 GB MimicLabs scene
cache, and every job that then missed re-downloaded the scenes from Google Drive until
Drive rate-limited the runners. Pip is now cached explicitly: saved on pushes to
`main`, restore-only everywhere else.
"""

from pathlib import Path
from typing import Any

import yaml

CI_WORKFLOW = Path(__file__).resolve().parents[1] / ".github" / "workflows" / "ci.yml"


class CiSteps:
    @staticmethod
    def jobs() -> dict[str, Any]:
        return yaml.safe_load(CI_WORKFLOW.read_text())["jobs"]

    @staticmethod
    def pip_steps(*, job: dict[str, Any]) -> list[dict[str, Any]]:
        return [
            step
            for step in job.get("steps", [])
            if str(step.get("uses", "")).startswith("actions/cache")
            and "pip" in str(step.get("with", {}).get("path", ""))
        ]


def test_setup_python_never_saves_its_own_pip_cache() -> None:
    for name, job in CiSteps.jobs().items():
        for step in job.get("steps", []):
            if str(step.get("uses", "")).startswith("actions/setup-python"):
                assert "cache" not in step.get("with", {}), name


def test_every_python_job_restores_pip_and_saves_it_only_from_main() -> None:
    for name, job in CiSteps.jobs().items():
        if not any(
            str(step.get("uses", "")).startswith("actions/setup-python")
            for step in job.get("steps", [])
        ):
            continue
        steps = CiSteps.pip_steps(job=job)
        saving = [step for step in steps if step["uses"].startswith("actions/cache@")]
        restoring = [step for step in steps if step["uses"].startswith("actions/cache/restore@")]
        assert len(saving) == 1 and len(restoring) == 1, name
        assert saving[0]["if"] == "github.event_name == 'push'", name
        assert restoring[0]["if"] == "github.event_name != 'push'", name
        assert saving[0]["with"]["key"] == restoring[0]["with"]["key"], name
