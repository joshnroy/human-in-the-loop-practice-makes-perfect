"""`MimicLabsAssets.ensure`: the one-download guarantee `pytest -n` workers rely on."""

import threading
import time
from pathlib import Path

import pytest

from .mimiclabs_assets import DISABLE_VAR, MimicLabsAssets


class SlowDownload:
    """Stands in for KINDER's downloader: slow enough that unsynchronised callers overlap."""

    def __init__(self, *, assets_dir: Path) -> None:
        self.assets_dir = assets_dir
        self.calls = 0

    def __call__(self) -> None:
        self.calls += 1
        time.sleep(0.3)
        (self.assets_dir / "mimiclabs_scenes" / "meshes").mkdir(parents=True)


def test_concurrent_workers_download_the_assets_exactly_once(
    *, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Four xdist workers starting on an asset-less checkout must not each fetch and
    extract into the same directory: KINDER's downloader `rmtree`s the scenes directory
    before moving the new tree in, under the feet of any worker already reading it."""
    monkeypatch.delenv(DISABLE_VAR, raising=False)
    download = SlowDownload(assets_dir=tmp_path)
    workers = [
        threading.Thread(
            target=MimicLabsAssets.ensure, kwargs={"assets_dir": tmp_path, "download": download}
        )
        for _ in range(4)
    ]
    for worker in workers:
        worker.start()
    for worker in workers:
        worker.join()
    assert download.calls == 1


def test_present_assets_are_never_downloaded(
    *, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv(DISABLE_VAR, raising=False)
    (tmp_path / "mimiclabs_scenes" / "meshes").mkdir(parents=True)
    download = SlowDownload(assets_dir=tmp_path)
    MimicLabsAssets.ensure(assets_dir=tmp_path, download=download)
    assert download.calls == 0


def test_the_disable_variable_is_honoured(
    *, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """CI sets it and supplies the assets from a cache step; never fetch behind its back."""
    monkeypatch.setenv(DISABLE_VAR, "1")
    download = SlowDownload(assets_dir=tmp_path)
    MimicLabsAssets.ensure(assets_dir=tmp_path, download=download)
    assert download.calls == 0
