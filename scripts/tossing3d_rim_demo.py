"""Replay a saved rim landing and record EES choosing and executing recovery."""

import argparse
import json
from pathlib import Path

import imageio.v2 as iio

from hitl_pmp.core.method.types import GroundAtom
from hitl_pmp.core.problem.tasks.types import Goal, Task
from hitl_pmp.environments.tossing3d.environment import Tossing3DEnvironment
from hitl_pmp.environments.tossing3d.layout import Tossing3DLayout
from hitl_pmp.environments.tossing3d.predicates import HOLDING, IN_BIN
from hitl_pmp.environments.tossing3d.skill_provider import Tossing3DSkillProvider
from hitl_pmp.methods.practice_makes_perfect.ees_method import EesMethod


class RimDemo:
    @staticmethod
    def capture(*, snapshot: Path, output_dir: Path) -> None:
        output_dir.mkdir(parents=True, exist_ok=True)
        env = Tossing3DEnvironment(layout=Tossing3DLayout.SAME_SIDE)
        try:
            env.set_substep_recording(enabled=True)
            env.hard_reset()
            env.drain_substep_frames()
            observed = env.restore_plain_snapshot(plain=json.loads(snapshot.read_text()))
            method = EesMethod(env=env, skill_provider=Tossing3DSkillProvider(env=env), seed=3)
            policy = method.get_practice_policy(
                task=Task(
                    initial_state=observed,
                    goal=Goal(
                        atoms=frozenset({GroundAtom(predicate=IN_BIN, objects=(env.cube, env.bin))})
                    ),
                )
            )
            action = policy(observed)
            with iio.get_writer(
                output_dir / "rim-recovery.mp4", fps=env.backend().render_fps()
            ) as video:
                first = env.backend().render()
                for _ in range(40):
                    video.append_data(first)
                retrieved = env.take_action(action=action.action)
                for frame in env.drain_substep_frames():
                    video.append_data(frame)
                final = env.backend().render()
                for _ in range(40):
                    video.append_data(final)
            result = {
                "source_snapshot": str(snapshot),
                "selected_skill": action.label,
                "holding": HOLDING.holds(retrieved, (env.robot, env.cube)),
                "cube_z": retrieved.get(obj=env.cube, feature_name="z"),
                "error": env.last_skill_error(),
                "human_resets": 0,
            }
            (output_dir / "result.json").write_text(json.dumps(result, indent=2) + "\n")
            print(json.dumps(result), flush=True)
            if not result["holding"] or result["cube_z"] <= 0.3:
                raise RuntimeError("Rim recovery did not grasp and lift the cube")
        finally:
            env.close()

    @staticmethod
    def main() -> None:
        parser = argparse.ArgumentParser(description=__doc__)
        parser.add_argument("--snapshot", type=Path, required=True)
        parser.add_argument("--output-dir", type=Path, required=True)
        args = parser.parse_args()
        RimDemo.capture(snapshot=args.snapshot, output_dir=args.output_dir)


if __name__ == "__main__":
    RimDemo.main()
