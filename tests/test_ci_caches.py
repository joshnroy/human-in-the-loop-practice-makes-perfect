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


MIRROR_TAG = "mimiclabs-lab2-v1"
MIRROR_SHA256 = "f1185bfde2a1ff09fff3b1f7372de0fe8ec61ce27546cdfca47c77524110d4a0"


def test_no_ci_step_downloads_mimiclabs_from_google_drive() -> None:
    # KINDER's own downloader fetches ~2 GB through gdown, which Drive rate-limits on CI.
    assert "download_mimiclabs_assets" not in CI_WORKFLOW.read_text()


def test_a_mimiclabs_cache_miss_fills_from_the_checksummed_lab2_mirror() -> None:
    jobs = [job for job in CiSteps.jobs().values() if "steps" in job]
    filled = 0
    for job in jobs:
        steps = job["steps"]
        if not any(step.get("id") == "cache-assets" for step in steps):
            continue
        (fill,) = [step for step in steps if MIRROR_TAG in str(step.get("run", ""))]
        assert fill["if"] == "steps.cache-assets.outputs.cache-hit != 'true'"
        assert f"{MIRROR_SHA256}  " in fill["run"] and "sha256sum -c" in fill["run"]
        filled += 1
    assert filled == 2  # The Tossing3D and scripts jobs, the two that build MimicLabs scenes.
