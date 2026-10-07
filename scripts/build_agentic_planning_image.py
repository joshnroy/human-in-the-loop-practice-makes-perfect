"""Prepare/build an offline PyBullet extension of the existing isolated CLI image."""

import argparse
import hashlib
import json
import shutil
import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path


class PlanningImageBuilder:
    """Copy only an audited binary and referenced robot geometry, never packages."""

    @staticmethod
    def prepare(*, context: Path, extension: Path, urdf: Path, base_image: str) -> None:
        if not extension.name.startswith("pybullet.cpython-311-") or extension.suffix != ".so":
            raise ValueError("The strict image requires a CPython 3.11 PyBullet extension")
        if not base_image.startswith("sha256:"):
            raise ValueError("Pin the existing strict image by its immutable SHA256 ID")
        context.mkdir(parents=True, exist_ok=False)
        shutil.copyfile(extension, context / extension.name)
        assets = context / "robot"
        assets.mkdir()
        tree = ET.parse(urdf)
        hashes = {extension.name: hashlib.sha256(extension.read_bytes()).hexdigest()}
        for mesh in tree.getroot().findall(".//mesh"):
            source_name = mesh.attrib["filename"]
            prefix = "package://kortex_description/"
            if not source_name.startswith(prefix):
                raise ValueError("Unexpected robot mesh source")
            relative = Path(source_name.removeprefix(prefix))
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError("Robot mesh path escapes its approved asset bundle")
            source = urdf.parent / relative
            if not source.resolve().is_relative_to(urdf.parent.resolve()):
                raise ValueError("Robot mesh symlink escapes the asset bundle")
            destination = assets / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, destination)
            mesh.set("filename", relative.as_posix())
            hashes[f"robot/{relative.as_posix()}"] = hashlib.sha256(source.read_bytes()).hexdigest()
        tree.write(assets / "gen3_7dof.urdf", encoding="utf-8", xml_declaration=True)
        root = Path(__file__).resolve().parents[1]
        for name in ("planning_scene.py", "planning_mcp.py"):
            source = root / "src/hitl_pmp/agentic_runtime" / name
            shutil.copyfile(source, context / name)
            hashes[name] = hashlib.sha256(source.read_bytes()).hexdigest()
        hashes["robot/gen3_7dof.urdf"] = hashlib.sha256(
            (assets / "gen3_7dof.urdf").read_bytes()
        ).hexdigest()
        (context / "manifest.json").write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "allowed_modules": ["pybullet"],
                    "base_image": base_image,
                    "sha256": hashes,
                    "model": "snapshot-only arm kinematics and primitive collision geometry",
                    "forbidden_packages": ["mujoco", "kinder", "kinder_models", "pybullet_helpers"],
                },
                indent=2,
            )
        )
        (context / "Dockerfile").write_text(
            f"FROM {base_image}\n"
            f"COPY {extension.name} /opt/robocode-strict/lib/python3.11/site-packages/\n"
            "COPY robot /opt/hitl-planning/robot\n"
            "COPY planning_scene.py planning_mcp.py manifest.json /opt/hitl-planning/\n"
        )

    @staticmethod
    def main() -> None:
        parser = argparse.ArgumentParser(description=__doc__)
        parser.add_argument("--context", type=Path, required=True)
        parser.add_argument("--extension", type=Path, required=True)
        parser.add_argument("--urdf", type=Path, required=True)
        parser.add_argument("--base-image", required=True)
        parser.add_argument("--tag", required=True)
        parser.add_argument("--prepare-only", action="store_true")
        args = parser.parse_args()
        PlanningImageBuilder.prepare(
            context=args.context,
            extension=args.extension,
            urdf=args.urdf,
            base_image=args.base_image,
        )
        if not args.prepare_only:
            subprocess.run(
                [
                    "docker",
                    "--host",
                    "unix:///var/run/docker.sock",
                    "build",
                    "--network",
                    "none",
                    "--pull=false",
                    "--tag",
                    args.tag,
                    str(args.context),
                ],
                check=True,
            )


if __name__ == "__main__":
    PlanningImageBuilder.main()
