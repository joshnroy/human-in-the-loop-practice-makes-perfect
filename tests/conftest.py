"""Session-wide setup shared by every test package.

## Why the MimicLabs assets are fetched here rather than left to KINDER

`kinder.make()` downloads ~1 GB of scene assets into its own checkout the first time a
scene is built. Under `pytest -n`, every worker reaches that first build at once, and
KINDER's check-then-download is not atomic, so each worker fetches the same archive and
`rmtree`s the scenes directory under the others. This fetches them once, under a file
lock, before any test runs. It is a no-op when the assets are present, when KINDER is not
installed, and when `DISABLE_AUTO_DYNAMIC3D_SCENES_DOWNLOAD` is set (as on CI, which
restores them from a cache step instead).
"""

import importlib.util
from pathlib import Path

import pytest

from .ci_shard import CiShard
from .mimiclabs_assets import MimicLabsAssets


@pytest.fixture(scope="session", autouse=True)
def _mimiclabs_assets_downloaded_once() -> None:
    # `find_spec` locates the package without importing it: importing `kinder` here
    # would pull MuJoCo into every worker, including those running only offline tests.
    spec = importlib.util.find_spec("kinder")
    if spec is None or not spec.submodule_search_locations:
        return

    def download() -> None:
        from kinder.utils import download_mimiclabs_assets

        download_mimiclabs_assets(non_interactive=True)

    package_root = Path(next(iter(spec.submodule_search_locations)))
    MimicLabsAssets.ensure(
        assets_dir=package_root / "envs" / "dynamic3d" / "models" / "assets", download=download
    )


def pytest_addoption(parser: pytest.Parser) -> None:  # noqa: PLR0917 - pluggy passes hook args positionally
    CiShard.add_options(parser=parser)


# Last, so a shard splits only what `-m` and `--ignore` left: an even split of the tests
# that will actually run, not of ones about to be deselected anyway.
@pytest.hookimpl(trylast=True)
def pytest_collection_modifyitems(  # noqa: PLR0917 - pluggy passes hook args positionally
    config: pytest.Config, items: list[pytest.Item]
) -> None:
    CiShard.apply(config=config, items=items)
