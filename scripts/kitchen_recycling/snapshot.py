"""Freeze worktree files with explicit canonical dependency pins, never dirty root code."""

import argparse
import hashlib
import io
import json
import subprocess
import tarfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
MODULES = ("reference/kindergarden", "reference/kinder-baselines")
CANONICAL = {
    "reference/kindergarden": "490831fa54a40c4b2b4570d45bb08e3dd63c24ef",
    "reference/kinder-baselines": "032b42634a015c7f710667b3a791836fa839f6ff",
}


def git(*, root: Path, arguments: list[str]) -> bytes:
    return subprocess.check_output(["git", "-C", str(root), *arguments])


def sha(*, path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--original-root", type=Path, required=True)
    parser.add_argument("--dependency-root", type=Path, required=True)
    parser.add_argument("--include-assets", action="store_true")
    parser.add_argument("--downward", type=Path)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    files = {}
    paths = []
    for subtree in ("src", "configs", "scripts/kitchen_recycling", "tests"):
        paths.extend((REPO / subtree).rglob("*"))
    paths += [
        REPO / name
        for name in ("pyproject.toml", "README.md", "LICENSE", "docs/kitchen-recycling3d.md")
    ]
    paths = sorted({
        path
        for path in paths
        if path.is_file()
        and "__pycache__" not in path.parts
        and path.suffix != ".pyc"
        and not any(part.endswith(".egg-info") for part in path.parts)
    })
    modules = {}
    with tarfile.open(args.out / "source.tar.gz", "w:gz", compresslevel=1) as archive:
        for path in paths:
            if path.is_symlink():
                raise ValueError(f"Source symlink is not portable: {path}")
            relative = path.relative_to(REPO).as_posix()
            content = path.read_bytes()
            files[relative] = hashlib.sha256(content).hexdigest()
            info = archive.gettarinfo(str(path), arcname=relative)
            archive.addfile(info, io.BytesIO(content))
        changed = [
            str(path.relative_to(REPO))
            for path in paths
            if sha(path=path) != files[str(path.relative_to(REPO))]
        ]
        if changed:
            raise RuntimeError(f"Source changed during copy; discard and rebuild: {changed}")
        for module in MODULES:
            revision = CANONICAL[module]
            actual = (
                git(root=args.dependency_root / module, arguments=["rev-parse", "HEAD"])
                .decode()
                .strip()
            )
            if actual != revision:
                raise ValueError(f"Expected pr-publish-20261027 pin for {module}: {revision}")
            modules[module] = revision
            content = git(
                root=args.dependency_root / module, arguments=["archive", "--format=tar", revision]
            )
            with tarfile.open(fileobj=io.BytesIO(content)) as upstream:
                for member in upstream:
                    if member.issym() or member.islnk():
                        raise ValueError(f"Review upstream link before packaging: {member.name}")
                    stream = upstream.extractfile(member) if member.isfile() else None
                    member.name = f"{module}/{member.name}"
                    if stream:
                        data = stream.read()
                        files[member.name] = hashlib.sha256(data).hexdigest()
                        archive.addfile(member, io.BytesIO(data))
                    else:
                        archive.addfile(member)
    if args.include_assets:
        relative = "reference/kindergarden/src/kinder/envs/dynamic3d/models/assets/mimiclabs_scenes"
        assets = args.original_root / relative
        with tarfile.open(args.out / "assets.tar", "w") as archive:
            for directory in ("meshes", "textures"):
                if not (assets / directory).is_dir():
                    raise FileNotFoundError(assets / directory)
                for path in sorted((assets / directory).rglob("*")):
                    if path.is_file():
                        if path.is_symlink():
                            raise ValueError(f"Asset symlink requires review: {path}")
                        name = f"{relative}/{path.relative_to(assets)}"
                        files[name] = sha(path=path)
                        archive.add(path, arcname=name, recursive=False)
    if args.downward:
        revision = git(root=args.downward, arguments=["rev-parse", "HEAD"]).decode().strip()
        content = git(root=args.downward, arguments=["archive", "--prefix=downward/", revision])
        (args.out / "downward.tar").write_bytes(content)
        modules["downward"] = revision
    metadata = {
        "worktree": str(REPO),
        "head": git(root=REPO, arguments=["rev-parse", "HEAD"]).decode().strip(),
        "status": git(root=REPO, arguments=["status", "--short"]).decode(),
        "modules": modules,
        "files": files,
        "dependency_source": str(args.dependency_root),
        "dependency_pin_basis": "pr-publish-20261027, explicitly verified; not worktree gitlinks",
        "archives": {path.name: sha(path=path) for path in args.out.glob("*.tar*")},
        "assets_included": args.include_assets,
        "status_note": "Snapshot only; not a launch authorization or a dependency validation",
    }
    (args.out / "snapshot.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps({"files": len(files), "archives": metadata["archives"]}))


if __name__ == "__main__":
    main()
