"""Download KINDER's MimicLabs scene assets at most once per checkout, across processes."""

import hashlib
import os
import tempfile
from collections.abc import Callable
from pathlib import Path

from filelock import FileLock

DISABLE_VAR = "DISABLE_AUTO_DYNAMIC3D_SCENES_DOWNLOAD"


class MimicLabsAssets:
    @staticmethod
    def ensure(*, assets_dir: Path, download: Callable[[], None]) -> None:
        """Run `download` iff the scene meshes are missing.

        The check-and-download is not atomic in KINDER (it tests for `meshes/` and then
        unpacks file by file), so it happens here under an inter-process lock, held for the
        whole download: a worker that finds the lock taken waits for a complete tree
        rather than racing into a half-extracted one.
        """
        if MimicLabsAssets.disabled():
            return
        with FileLock(MimicLabsAssets.lock_path(assets_dir=assets_dir)):
            if not (assets_dir / "mimiclabs_scenes" / "meshes").exists():
                download()

    @staticmethod
    def lock_path(*, assets_dir: Path) -> Path:
        """Outside the KINDER tree: a file in it would mark the submodule dirty, which
        `ConfigSnapshot` records, and the downloader `rmtree`s the scenes directory -- an
        unlinked lock file no longer excludes a process that opens the path afresh."""
        digest = hashlib.sha256(str(assets_dir.resolve()).encode()).hexdigest()[:16]
        return Path(tempfile.gettempdir()) / f"hitl-pmp-mimiclabs-{digest}.lock"

    @staticmethod
    def disabled() -> bool:
        return os.environ.get(DISABLE_VAR, "").strip().lower() in {"1", "true", "yes", "on"}
