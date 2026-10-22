"""Freeze everything a RoboCode (Full-Agentic, LLLL) shelf launch runs from.

    python -m scripts.robocode_prepare --root DIR --ref REF [--robocode DIR] [--image IMG]

Writes, under `--root` (refusing if `source/` exists):

- `source/repo`: `git archive` of `--ref`, with `reference/kindergarden` and
  `reference/kinder-baselines` copied in (hard links where possible) from checkouts that
  must sit exactly at the gitlinks that commit pins, so the tree a run imports is the
  tree it declares, and `ShelfLaunch.require_isolated` passes;
- `source/robocode`: the frozen RoboCode bootstrap, refused unless every file the frozen
  repo's `full_agentic/bootstrap_provenance.json` pins matches (135/135);
- `sandbox.json`: the coder and execution sandbox settings, image resolved to its ID;
- `source-versions.json`: every commit and the provenance count;
- `launch.sh`: the launch, which runs only with `--go` appended, on Josh's go.

The launch passes `--hours LAUNCH_HOURS`, the limit Josh chose; the launcher itself has no
default. A dry run passes its own budgets after it, e.g.
`launch.sh --size dry --hours 1 --model-budget 3 --go`, which override (argparse keeps the
last value).
"""

import argparse
import hashlib
import json
import os
import shlex
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
# Josh, 2026-10-22: one RoboCode seed, at most 6 h of adaptation wall-clock.
LAUNCH_HOURS = 6
DEFAULT_IMAGE = "hitl-agentic-strict:20261002"
DEFAULT_TOKEN_FILE = Path.home() / ".bashrc"
KINDER_CHECKOUTS = ("kindergarden", "kinder-baselines")
PROVENANCE = Path("src/hitl_pmp/full_agentic/bootstrap_provenance.json")


class RoboCodePrepare:
    """A static-method container, never instantiated."""

    @staticmethod
    def matching(*, robocode: Path, pinned: dict[str, str]) -> tuple[int, int]:
        """(files matching their pinned sha256, files pinned)."""
        ok = sum(
            (robocode / path).is_file()
            and hashlib.sha256((robocode / path).read_bytes()).hexdigest() == digest
            for path, digest in pinned.items()
        )
        return ok, len(pinned)

    @staticmethod
    def verify_robocode(*, robocode: Path, pinned: dict[str, str]) -> None:
        ok, total = RoboCodePrepare.matching(robocode=robocode, pinned=pinned)
        if ok != total:
            raise SystemExit(f"refusing: RoboCode at {robocode} matches {ok}/{total} pinned files")

    @staticmethod
    def require_pin(*, name: str, pinned: str, checked_out: str) -> None:
        if pinned != checked_out:
            raise SystemExit(
                f"refusing: the frozen commit pins {name} at {pinned}, "
                f"but the checkout to copy is at {checked_out}"
            )

    @staticmethod
    def sandbox_settings(*, root: Path, image: str) -> dict[str, Any]:
        return {
            "image": image,
            "robocode_checkout": str(root / "source" / "robocode"),
            "container_python": "/opt/robocode-strict/bin/python",
            "timeout_seconds": 120,
            "artifact_dir": str(root / "execution"),
            "seed": 0,
            "memory_mb": 4096,
            "coder_memory_mb": 16384,
            "cpus": 2.0,
            "resource_pool_dir": str(root / "resource-pool"),
            "cgroup_parent": None,
        }

    @staticmethod
    def launch_script(*, root: Path, token_file: Path) -> str:
        repo = root / "source" / "repo"
        command = shlex.join([
            str(repo / "scripts" / "with_env.sh"), "python",
            "-m", "scripts.sweepoffshelf3d_launch", "launch",
            "--arms", "robocode",
            "--hours", str(LAUNCH_HOURS),
            "--results-root", str(root / "results"),
            "--sandbox-settings", str(root / "sandbox.json"),
            "--token-file", str(token_file),
        ])  # fmt: skip
        return (
            "#!/bin/bash\n"
            "# RoboCode on SweepOffShelf3D, written by scripts/robocode_prepare.py.\n"
            "# Starts nothing without --go, which is Josh's go: launch.sh --go\n"
            "set -euo pipefail\n"
            f"cd {shlex.quote(str(repo))}\n"
            f'{command} "$@"\n'
        )

    @staticmethod
    def git(*args: str, cwd: Path) -> str:
        return subprocess.run(
            ["git", *args], cwd=cwd, capture_output=True, text=True, check=True
        ).stdout.strip()

    @staticmethod
    def copy_checkout(*, source: Path, destination: Path) -> None:
        """Hard links where the filesystem allows, so a 2 GB asset tree costs nothing."""

        def link(src: str, dst: str) -> None:  # noqa: PLR0917 -- shutil's callback
            try:
                os.link(src, dst)
            except OSError:
                shutil.copy2(src, dst)

        if destination.exists():
            destination.rmdir()
        shutil.copytree(
            source, destination, copy_function=link, ignore=shutil.ignore_patterns(".git")
        )

    @staticmethod
    def image_id(*, image: str) -> str:
        return subprocess.run(
            ["docker", "image", "inspect", "--format", "{{.Id}}", image],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()

    @staticmethod
    def parse(*, argv: list[str]) -> argparse.Namespace:
        parser = argparse.ArgumentParser(
            description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
        )
        main_checkout = Path(
            RoboCodePrepare.git("rev-parse", "--path-format=absolute", "--git-common-dir", cwd=REPO)
        ).parent
        parser.add_argument("--root", type=Path, required=True)
        parser.add_argument("--ref", required=True, help="The commit to freeze.")
        parser.add_argument(
            "--robocode",
            type=Path,
            default=main_checkout / "artifacts/full-agent-20261006/source/robocode",
        )
        parser.add_argument("--image", default=DEFAULT_IMAGE)
        parser.add_argument("--token-file", type=Path, default=DEFAULT_TOKEN_FILE)
        for name in KINDER_CHECKOUTS:
            parser.add_argument(f"--{name}", type=Path, default=REPO / "reference" / name)
        return parser.parse_args(argv)

    @staticmethod
    def main(*, argv: list[str]) -> int:
        args = RoboCodePrepare.parse(argv=argv)
        root = args.root.resolve()
        source = root / "source"
        if source.exists():
            raise SystemExit(f"refusing: {source} exists")
        sha = RoboCodePrepare.git("rev-parse", "--verify", f"{args.ref}^{{commit}}", cwd=REPO)
        repo = source / "repo"
        repo.mkdir(parents=True)
        archive = subprocess.run(["git", "archive", sha], cwd=REPO, capture_output=True, check=True)
        subprocess.run(["tar", "-x", "-C", str(repo)], input=archive.stdout, check=True)
        versions: dict[str, Any] = {"repo_ref": args.ref, "repo_commit": sha}
        for name in KINDER_CHECKOUTS:
            checkout = getattr(args, name.replace("-", "_")).resolve()
            pinned = RoboCodePrepare.git("ls-tree", sha, f"reference/{name}", cwd=REPO).split()[2]
            RoboCodePrepare.require_pin(
                name=name,
                pinned=pinned,
                checked_out=RoboCodePrepare.git("rev-parse", "HEAD", cwd=checkout),
            )
            if RoboCodePrepare.git("status", "--porcelain", "--untracked-files=no", cwd=checkout):
                raise SystemExit(f"refusing: {checkout} has uncommitted changes")
            RoboCodePrepare.copy_checkout(source=checkout, destination=repo / "reference" / name)
            versions[f"{name.replace('-', '_')}_commit"] = pinned
        provenance = json.loads((repo / PROVENANCE).read_text())
        shutil.copytree(args.robocode, source / "robocode", symlinks=True)
        pinned_files = provenance["robocode_source"]
        RoboCodePrepare.verify_robocode(robocode=source / "robocode", pinned=pinned_files)
        ok, total = RoboCodePrepare.matching(robocode=source / "robocode", pinned=pinned_files)
        image = RoboCodePrepare.image_id(image=args.image)
        (root / "resource-pool").mkdir(exist_ok=True)
        (root / "sandbox.json").write_text(
            json.dumps(RoboCodePrepare.sandbox_settings(root=root, image=image), indent=2) + "\n"
        )
        versions |= {
            "image": args.image,
            "image_id": image,
            "robocode_origin": provenance["origin"],
            "robocode_files_matching_provenance": f"{ok}/{total}",
            "launch_hours": LAUNCH_HOURS,
        }
        (root / "source-versions.json").write_text(json.dumps(versions, indent=2) + "\n")
        launch = root / "launch.sh"
        launch.write_text(RoboCodePrepare.launch_script(root=root, token_file=args.token_file))
        launch.chmod(0o755)
        print(f"frozen {args.ref} @ {sha[:9]}; robocode {ok}/{total}; {launch}")
        return 0


if __name__ == "__main__":
    raise SystemExit(RoboCodePrepare.main(argv=sys.argv[1:]))
