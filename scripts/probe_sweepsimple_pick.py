"""One explicit native-scene pickup feasibility probe, not a learning run."""

import argparse
import hashlib
import json
import shutil
import sys
import time
from pathlib import Path


class PickupProbe:
    @staticmethod
    def run() -> None:
        parser = argparse.ArgumentParser()
        parser.add_argument("--tag", required=True)
        parser.add_argument("--seed", type=int, default=0)
        parser.add_argument("--pick-distance", type=float, default=0.7)
        parser.add_argument("--sweep-distance", type=float, default=0.55)
        parser.add_argument("--sweep-angle", type=float, default=0.0)
        parser.add_argument("--cube-order", type=int, nargs=5, default=[0, 1, 2, 3, 4])
        parser.add_argument(
            "--forward-budget", type=int, choices=(0, 10), default=0,
            help="Count failed skill calls within the approved 10-action task budget; no resets.",
        )
        parser.add_argument("--pick-only", action="store_true")
        parser.add_argument(
            "--full-cycle",
            action="store_true",
            help="Check all native goals, individual reverse sweeps, and existing return skills.",
        )
        parser.add_argument("--grasp-offset", type=float)
        parser.add_argument("--grasp-height", type=float, default=0.0)
        parser.add_argument("--grasp-mode", choices=("handle", "blade"), default="handle")
        parser.add_argument("--tilt-limit", type=float, default=1.1)
        parser.add_argument("--stroke-length", type=float, default=0.10)
        parser.add_argument("--contact-step", type=float, default=0.003)
        parser.add_argument("--floor-clearance", type=float, default=0.001)
        parser.add_argument("--narrow-contact", action="store_true")
        resume = parser.add_mutually_exclusive_group()
        resume.add_argument(
            "--resume-pick", help="Development replay tag; excluded from end-to-end readiness"
        )
        resume.add_argument(
            "--resume-final",
            help="Failed probe final-state tag; development replay, excluded from readiness",
        )
        parser.add_argument("--grasp-yaw-offset", type=float, default=0.0)
        args = parser.parse_args()
        if sorted(args.cube_order) != list(range(5)):
            parser.error("--cube-order must contain every native cube index exactly once")
        if args.full_cycle and args.pick_only:
            parser.error("--full-cycle cannot be combined with --pick-only")
        if args.forward_budget and not args.full_cycle:
            parser.error("--forward-budget requires --full-cycle")
        resume_tag = args.resume_pick or args.resume_final
        resume_phase = (
            "final_recorded_state"
            if args.resume_final
            else "picked_state"
            if args.resume_pick
            else None
        )
        import faulthandler

        faulthandler.enable()
        faulthandler.dump_traceback_later(120, repeat=True)
        root = Path(__file__).resolve().parents[1]
        sys.path.insert(0, str(root / "reference/kindergarden/src"))
        sys.path.insert(0, str(root / "reference/kinder-baselines/kinder-models/src"))
        from hitl_pmp.environments.sweep_simple3d.controllers import FloorPrimitives
        from hitl_pmp.environments.sweep_simple3d.session import SweepSimpleSession

        output = root / "scratchpad/sweepsimple3d" / args.tag
        if output.exists():
            raise FileExistsError(output)
        output.mkdir(parents=True, exist_ok=True)
        source_dir = output / "source"
        source_dir.mkdir()
        shutil.copy2(__file__, source_dir / "probe_driver.py")
        hashes = {}
        for source in [
            *root.glob("src/hitl_pmp/environments/sweep_simple3d/*.py"),
            *[
                root / "src/hitl_pmp/environments/sweep_drawer3d" / name
                for name in ("primitives.py", "session.py", "motion.py", "planning_scene.py")
            ],
        ]:
            label = source.parent.name + "__" + source.name
            shutil.copy2(source, source_dir / label)
            hashes[label] = hashlib.sha256(source.read_bytes()).hexdigest()
        (output / "probe_manifest.json").write_text(
            json.dumps(
                {
                    "arguments": vars(args),
                    "source_sha256": hashes,
                    "resume_phase": resume_phase,
                    "end_to_end_native_start": resume_tag is None,
                },
                indent=2,
            )
        )
        session = SweepSimpleSession(
            seed=args.seed, log_path=output / "state.jsonl", replay_path=output / "replay.jsonl"
        )
        initial_state = session.state.copy()
        if resume_tag:
            PickupProbe.restore_recorded(
                session=session,
                source=root / "scratchpad/sweepsimple3d" / resume_tag,
                final=args.resume_final is not None,
            )
        primitive = FloorPrimitives.create(
            session=session, distance=args.pick_distance, heading_offset=0.0
        )
        primitive.scene.max_tool_tilt = args.tilt_limit
        primitive.contact_stroke_length = args.stroke_length
        primitive.contact_step = args.contact_step
        primitive.floor_clearance = args.floor_clearance
        primitive.narrow_contact = args.narrow_contact
        if args.grasp_mode == "blade":
            import mujoco
            import numpy as np

            def blade_geometry(self):  # noqa: PLR0917 -- installed as a bound probe method
                model = self.session.mj_model
                body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "wiper_0")
                blade = max(
                    (g for g in range(model.ngeom) if model.geom_bodyid[g] == body),
                    key=lambda g: float(model.geom_size[g][0]),
                )
                return blade, 0

            FloorPrimitives.wiper_handle_geometry = blade_geometry
            FloorPrimitives.wiper_approach_angles = lambda self: (0.0, 0.4, 0.7)
            FloorPrimitives.wiper_grasp_yaw = lambda self, *, axis: float(
                np.arctan2(axis[1], axis[0]) + np.pi / 2
            )
        if args.grasp_yaw_offset:
            original_yaw = FloorPrimitives.wiper_grasp_yaw
            FloorPrimitives.wiper_grasp_yaw = lambda self, *, axis: (
                original_yaw(self, axis=axis) + args.grasp_yaw_offset
            )
        if args.grasp_height:
            import numpy as np

            FloorPrimitives.wiper_grasp_point = lambda self, *, center, axis, along: (
                center + along * axis + np.array([0.0, 0.0, args.grasp_height])
            )
        if args.grasp_offset is not None:
            FloorPrimitives.wiper_grasp_offsets = lambda self: (args.grasp_offset,)
        import mujoco

        started = time.monotonic()
        session.begin(name="PickFloorWiper", kind="PickFloorWiper", phase="feasibility")
        error = ""
        error_traceback = ""
        success = False
        cycle = {
            "stages": [],
            "native_goal_success": None,
            "start_distribution_shared": None,
            "start_distribution_strict": None,
            "wiper_start_support_tolerance_m": 0.01,
            "end_to_end_native_start": resume_tag is None,
        }
        if args.full_cycle:
            cycle["stages"].append({
                "name": "PickFloorWiper",
                "cube": None,
                "success": False,
                "replayed": resume_tag is not None,
                "resume_phase": resume_phase,
                "tick_start": session.ticks,
            })
        try:
            note = (
                f"Recorded {resume_phase} continuation; not an end-to-end native-start trial"
                if resume_tag
                else primitive.recover_wiper()
            )
            primitive.require_handle(phase="verified pickup or recorded continuation")
            if args.full_cycle:
                cycle["stages"][-1].update(success=True, tick_end=session.ticks, note=note)
            import mujoco

            contacts = []
            for contact in session.mj_data.contact:
                names = [
                    mujoco.mj_id2name(
                        session.mj_model, mujoco.mjtObj.mjOBJ_BODY, session.mj_model.geom_bodyid[g]
                    )
                    for g in (contact.geom1, contact.geom2)
                ]
                if "wiper_0" in names:
                    contacts.append(names)
            print(
                json.dumps({
                    "phase": "after_pick",
                    "contacts": contacts,
                    "pads": [p.tolist() for p in primitive.pad_centers()],
                    "tool": session.position(name="wiper_0").tolist(),
                }),
                flush=True,
            )
            if not args.pick_only:
                session.end(success=True, note=note)
                session.begin(name="SweepCubeToGoal", kind="SweepCubeToGoal", phase="feasibility")
                if args.full_cycle:
                    cycle["stages"].append({
                        "name": "SweepCubeToGoal",
                        "cube": f"cube_{args.cube_order[0]}",
                        "success": False,
                        "tick_start": session.ticks,
                    })
                from hitl_pmp.environments.sweep_drawer3d.motion import ExecutionError

                sweep_error = None
                try:
                    note = primitive.sweep_cube(
                        cube=f"cube_{args.cube_order[0]}",
                        region="sweep_region",
                        distance=args.sweep_distance,
                        heading_offset=args.sweep_angle,
                    )
                except ExecutionError as exc:
                    if not args.forward_budget:
                        raise
                    sweep_error = repr(exc)
                    note = sweep_error
                if args.full_cycle:
                    from hitl_pmp.environments.sweep_simple3d.regions import SimpleRegions

                    attained = SimpleRegions.contains(
                        session=session, name=f"cube_{args.cube_order[0]}", region="sweep_region"
                    )
                    cycle["stages"][-1].update(
                        success=attained and sweep_error is None,
                        native_target_attained=attained,
                        error=sweep_error,
                        tick_end=session.ticks,
                        note=note,
                        native_counts=PickupProbe.native_counts(session=session),
                    )
                    if not attained and not args.forward_budget:
                        raise RuntimeError("Selected cube sweep did not attain its native goal")
                    session.end(success=attained and sweep_error is None, note=note)
                    PickupProbe.full_cycle(
                        session=session,
                        primitive=primitive,
                        initial_state=initial_state,
                        args=args,
                        report=cycle,
                    )
                    note = "All native cube goals and the shared validated start contract attained"
            success = True
        except Exception as exc:
            import traceback

            error = repr(exc)
            error_traceback = traceback.format_exc()
            print(error_traceback, flush=True)
            note = error
            if args.full_cycle and cycle["stages"] and not cycle["stages"][-1]["success"]:
                cycle["stages"][-1].update(error=error, tick_end=session.ticks)
        finally:
            if session._current is not None:
                session.end(success=success, note=note)
            result = {
                "success": success,
                "end_to_end_native_start": resume_tag is None,
                "resumed_pick": args.resume_pick,
                "resumed_final": args.resume_final,
                "resume_phase": resume_phase,
                "error": error,
                "traceback": error_traceback,
                "note": note,
                "elapsed": time.monotonic() - started,
                "ticks": session.ticks,
                "wiper_position": session.position(name="wiper_0").tolist(),
                "final_contacts": [
                    [
                        mujoco.mj_id2name(
                            session.mj_model,
                            mujoco.mjtObj.mjOBJ_BODY,
                            session.mj_model.geom_bodyid[g],
                        )
                        for g in (contact.geom1, contact.geom2)
                    ]
                    for contact in session.mj_data.contact
                    if any(
                        mujoco.mj_id2name(
                            session.mj_model,
                            mujoco.mjtObj.mjOBJ_BODY,
                            session.mj_model.geom_bodyid[g],
                        )
                        == "wiper_0"
                        for g in (contact.geom1, contact.geom2)
                    )
                ],
            }
            if args.full_cycle:
                cycle["final_native_counts"] = PickupProbe.native_counts(session=session)
                cycle["stages_succeeded"] = sum(bool(s["success"]) for s in cycle["stages"])
                cycle["stages_attempted"] = len(cycle["stages"])
                cycle["robot_actions_executed"] = sum(
                    not stage.get("skipped", False) and not stage.get("replayed", False)
                    for stage in cycle["stages"]
                    if not stage["name"].startswith("Validate")
                )
                result["full_cycle"] = cycle
            (output / "result.json").write_text(json.dumps(result, indent=2))
            print(json.dumps(result), flush=True)
            primitive.scene._sim.close()
            session.close()
            faulthandler.cancel_dump_traceback_later()

    @staticmethod
    def native_counts(*, session) -> dict[str, int]:
        from hitl_pmp.environments.sweep_simple3d.regions import SimpleRegions

        return {
            region: sum(
                SimpleRegions.contains(session=session, name=f"cube_{i}", region=region)
                for i in range(5)
            )
            for region in ("sweep_region", "blocks_init_region")
        }

    @staticmethod
    def full_cycle(*, session, primitive, initial_state, args, report) -> None:
        """Compose existing skills on the live session; no restoration or hidden pickup."""
        from hitl_pmp.environments.sweep_simple3d.environment import SweepSimpleEnvironment
        from hitl_pmp.environments.sweep_simple3d.regions import SimpleRegions
        from hitl_pmp.environments.sweep_simple3d.symbolic import SimpleSymbols

        env = SweepSimpleEnvironment(canonical_seed=session.seed)
        env._session, env._primitive, env._initial_state = session, primitive, initial_state
        env.current_state = env.observe()
        core = session.env.unwrapped._object_centric_env
        if args.forward_budget:
            report["forward_action_budget"] = args.forward_budget
            # The initial pickup and first sweep already consumed two actions.
            # Rotate through the remaining native subgoals; a failed skill stays
            # failed and consumes its action. No state restoration or hidden pick.
            order = args.cube_order[1:] + args.cube_order[:1]
            for slot in range(2, args.forward_budget):
                if core._check_goals():
                    break
                pending = [i for i in order if not SimpleRegions.contains(
                    session=session, name=f"cube_{i}", region="sweep_region"
                )]
                if not pending:
                    break
                i = pending[0]
                order = order[order.index(i) + 1:] + order[:order.index(i) + 1]
                try:
                    PickupProbe.cycle_action(
                        env=env, name="SweepCubeToGoal", cube=i, report=report,
                        params=(args.sweep_distance, args.sweep_angle),
                    )
                except RuntimeError as exc:
                    if "note" not in report["stages"][-1]:
                        raise
                    report["stages"][-1]["counted_failure"] = repr(exc)
                report["stages"][-1]["forward_action_index"] = slot + 1
            report["forward_robot_actions_executed"] = sum(
                stage["name"] in {"PickFloorWiper", "SweepCubeToGoal"}
                and not stage.get("skipped", False)
                for stage in report["stages"]
            )
            if report["forward_robot_actions_executed"] > args.forward_budget:
                raise RuntimeError("Readiness probe exceeded the approved forward action budget")
        else:
            for i in args.cube_order[1:]:
                PickupProbe.cycle_action(
                    env=env,
                    name="SweepCubeToGoal",
                    cube=i,
                    report=report,
                    params=(args.sweep_distance, args.sweep_angle),
                    already_satisfied=SimpleRegions.contains(
                        session=session, name=f"cube_{i}", region="sweep_region"
                    ),
                )
        report["native_goal_success"] = bool(core._check_goals())
        report["forward_native_counts"] = PickupProbe.native_counts(session=session)
        report["stages"].append({
            "name": "ValidateNativeGoal",
            "cube": None,
            "success": report["native_goal_success"],
            "tick_start": session.ticks,
            "tick_end": session.ticks,
            "native_counts": report["forward_native_counts"],
        })
        if not report["native_goal_success"]:
            raise RuntimeError("All-cube native goal check failed before recovery")
        for i in range(5):
            PickupProbe.cycle_action(
                env=env,
                name="SweepCubeToStart",
                cube=i,
                report=report,
                already_satisfied=SimpleRegions.contains(
                    session=session, name=f"cube_{i}", region="blocks_init_region"
                ),
            )
        for name, fact in (("PlaceWiperAtStart", "WiperHome"), ("ReturnRobotToStart", "RobotHome")):
            PickupProbe.cycle_action(
                env=env,
                name=name,
                cube=-1,
                report=report,
                already_satisfied=bool(
                    env.get_current_state().get(obj=SimpleSymbols.SCENE, feature_name=fact)
                    and env.get_current_state().get(
                        obj=SimpleSymbols.SCENE, feature_name="HandEmpty"
                    )
                ),
            )
        validation = SimpleRegions.validate(session=session)
        strict = dict(validation.checks)
        for _, name, region in core.task_config["initial_state"]:
            strict[name + ":region"] = SimpleRegions.contains(
                session=session, name=name, region=region, support_tolerance=0.0
            )
        hand_empty = bool(
            env.get_current_state().get(obj=SimpleSymbols.SCENE, feature_name="HandEmpty")
        )
        report["start_validation"] = validation.model_dump()
        report["strict_start_checks"] = strict
        report["start_distribution_shared"] = validation.valid
        report["start_distribution_strict"] = all(strict.values())
        report["hand_empty_after_return"] = hand_empty
        stage = {
            "name": "ValidateStartDistribution",
            "cube": None,
            "success": bool(validation.valid and hand_empty),
            "tick_start": session.ticks,
            "tick_end": session.ticks,
            "native_counts": PickupProbe.native_counts(session=session),
        }
        report["stages"].append(stage)
        if not stage["success"]:
            raise RuntimeError("Robot recovery failed the existing shared validated start contract")

    @staticmethod
    def cycle_action(
        *, env, name, cube, report, already_satisfied=False, params=(0.0, 0.0)
    ) -> None:
        import numpy as np

        session = env.session()
        stage = {
            "name": name,
            "cube": None if cube < 0 else f"cube_{cube}",
            "success": False,
            "skipped": already_satisfied,
            "tick_start": session.ticks,
        }
        report["stages"].append(stage)
        try:
            if not already_satisfied:
                env.take_action(
                    action=np.array([env.ACTION_NAMES.index(name), cube, *params], dtype=float)
                )
                observed = session._steps[-1]
                stage.update(success=bool(observed.success), note=observed.note)
                if not observed.success or observed.note:
                    stage["success"] = False
                    raise RuntimeError(f"{name}({cube}) failed: {observed.note}")
            else:
                stage.update(
                    success=True, note="Native target already satisfied; no action executed"
                )
        finally:
            stage.update(
                tick_end=session.ticks, native_counts=PickupProbe.native_counts(session=session)
            )

    @staticmethod
    def restore_pick(*, session, source: Path) -> None:
        """Compatibility entrypoint for picked-state diagnostic continuation."""
        PickupProbe.restore_recorded(session=session, source=source, final=False)

    @staticmethod
    def restore_recorded(*, session, source: Path, final: bool) -> None:
        """Restore recorded native joints for explicitly labeled controller debugging."""
        import mujoco

        source_manifest = json.loads((source / "probe_manifest.json").read_text())
        if source_manifest["arguments"]["seed"] != session.seed:
            raise ValueError("Replay and requested native seeds differ")
        records = [json.loads(line) for line in (source / "state.jsonl").open()]
        if final:
            result = json.loads((source / "result.json").read_text())
            if result.get("success") is not False:
                raise ValueError("Final-state continuation requires a completed failed probe")
            picked = next(
                (
                    record
                    for record in reversed(records)
                    if record.get("kind") == "tick" and "state" in record
                ),
                None,
            )
            if picked is None or picked["t"] != result["ticks"]:
                raise ValueError("Final native tick and failed result disagree")
        else:
            picked = next(
                (
                    record
                    for record in reversed(records)
                    if record.get("kind") == "tick" and record.get("step") == "PickFloorWiper"
                ),
                None,
            )
            subsequent_sweep = any(record.get("step") == "SweepCubeToGoal" for record in records)
            if picked is None or not subsequent_sweep:
                raise ValueError("Source does not establish a completed pickup followed by a sweep")
        state = session.state.copy()
        for name, values in picked["state"].items():
            obj = state.get_object_from_name(name)
            for key, value in zip(state.type_features[obj.type], values, strict=True):
                is_velocity = key.startswith("vel_") or key in {"vx", "vy", "vz", "wx", "wy", "wz"}
                state.set(obj, key, 0.0 if is_velocity else value)
        session.env.unwrapped._object_centric_env.set_state(state)
        session._state = state
        frame = None
        for line in (source / "replay.jsonl").open():
            record = json.loads(line)
            if record.get("t") == picked["t"]:
                frame = record
                break
        if frame is None:
            raise ValueError("Native joint replay frame is missing")
        precise = next(
            (
                record
                for record in reversed(records)
                if record.get("t") == picked["t"] and "native_qpos" in record
            ),
            None,
        )
        session.mj_data.qpos[:] = precise["native_qpos"] if precise is not None else frame["qpos"]
        session.mj_data.qvel[:] = 0
        mujoco.mj_forward(session.mj_model, session.mj_data)
        session._write(
            record={
                "kind": "development_replay_resume",
                "resume_phase": "final_recorded_state" if final else "picked_state",
                "qpos_source": "native_qpos" if precise is not None else "rounded_replay_qpos",
                "source": str(source),
                "source_tick": picked["t"],
                "source_manifest_sha256": hashlib.sha256(
                    (source / "probe_manifest.json").read_bytes()
                ).hexdigest(),
                "velocities": "zeroed; complete native velocities were not recorded",
                "end_to_end_native_start": False,
            }
        )
        session._tick()


if __name__ == "__main__":
    PickupProbe.run()
