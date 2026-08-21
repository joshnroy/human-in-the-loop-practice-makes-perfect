"""`Tossing3DEnvironment` -- KINDER's `Tossing3D` behind `core.Environment`.

A TidyBot++ mobile manipulator must get a cube from the floor to a goal region on the
far side of an immovable 5 m barrier. The base cannot pass the barrier, so the cube can
only get there through the air: the robot must **toss** it. A tossed cube cannot be
retrieved -- hit or miss, it ends up past the barrier and no skill brings it back. That
irreversibility is why this domain is here.

## One transition is one skill, not one control tick

KINDER steps at `control_frequency = 10 Hz` over a `0.0005 s` simulation timestep, i.e.
200 physics substeps per env step, and a single grasp takes ~70 of those env steps.
`core.Environment.take_action` therefore takes a *skill*:
`[skill_id, param_0, param_1, param_2, param_3]`. The controllers that turn that into
hundreds of 11- or 18-dimensional joint commands are upstream's, unmodified, behind
`KinderBackend`.

## Two skills, not three

`pick_cube` and `move_to_toss_location_and_toss` are the whole domain. The base move and
the throw used to be separate skills (`MoveToThrowPose`, `Toss`) with a predicate --
`RobotAtSuccessfulThrowPose` -- naming the pose between them; upstream composed them, so
that predicate has no skill to be the effect of and is gone. See `skills.py`.

## `set_state` can only restore an episode-initial state, and says so

Every other domain in this repo has a `core.State` that fully determines its dynamics, so
`set_state` is a total rewind. Here it is not: MuJoCo's `qpos`/`qvel` (contact state,
joint velocities, the arm's whole configuration) do not fit in a flat feature vector, and
this package deliberately does not smuggle a simulator handle into `State`. So the only
restorable states are the ones a `reset(seed=...)` produces, and `set_state` rebuilds the
scene from the seed carried in the state's own `scene` object.

Handing it a mid-episode state **raises** rather than silently restoring something else.
That is the honest behaviour: `PracticeLoop.reset_to_task` and
`Method.reset_environment` only ever pass a `Task.initial_state`, which is exactly the
restorable case, and anything that wants a general rewind is asking for something this
domain cannot do.
"""

from typing import Any, ClassVar

import numpy as np
from gymnasium.spaces import Box
from pydantic import BaseModel, ConfigDict, PrivateAttr

from hitl_pmp.core.problem.environment.environment import Environment
from hitl_pmp.core.problem.environment.types import Action, Object, State, Type

from .kinder_backend import ControllerRun, KinderBackend, KinderObservation
from .types import AbstractAtom, Tossing3DState


class Tossing3DSnapshot(BaseModel):
    """A rewind point: KINDER's own full state, plus the `core.State` that projected it.

    Both halves are needed. `kinder_state` is what actually puts the simulator back;
    `state` is what this environment reports afterwards, kept alongside so a restore does
    not have to re-derive it and cannot disagree with the rewind it just performed.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)

    kinder_state: Any
    state: State


class Tossing3DEnvironment(Environment):
    """Concrete `core.Environment` over a live KINDER `Tossing3D` simulator.

    Constructing one imports nothing -- `KinderBackend` is pure pydantic until its first
    `reset()` -- so this class, its predicates and its skills are importable, typecheckable
    and unit-testable on a machine with no MuJoCo. That is what lets CI run the offline
    half of this domain's tests without the optional `tossing3d` extra.

    `Type`s and singleton `Object`s stay `ClassVar`s: the scene's cast is fixed by
    upstream's task JSON and does not vary between two instances. Feature schemas are
    mostly a *subset* of KINDER's own -- every name below appears verbatim in
    `kinder/envs/dynamic3d/object_types.py` with two exceptions, both ours and both
    flagged where they are declared: `scene` (see `scene_type`), and the bin's six bbox
    features, which are the scored region's box rather than anything KINDER calls a bin
    feature (see `bin_type`).
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    # KINDER's `MujocoTidyBotRobotObjectType` carries 22 features; these are the four the
    # symbolic layer reads. `pos_gripper` is what upstream's own `HandEmpty`/`Holding`
    # classifiers key on.
    robot_type: ClassVar[Type] = Type(
        name="tossing3d_robot",
        feature_names=("pos_base_x", "pos_base_y", "pos_base_rot", "pos_gripper"),
    )
    # A subset of `MujocoMovableObjectType`'s 16. `bb_z` is the cube's own bounding-box
    # height, which upstream's `OnGround` uses to decide "resting on the floor" without a
    # hardcoded cube size.
    #
    # The full quaternion and all three extents are carried because upstream's `OnGround`
    # is a face-interchangeable test rather than the old `qx`/`qy`-near-zero one: it
    # composes the measured rotation with the cube's 24-element symmetry group, and it
    # decides whether an object *is* a cube by comparing `bb_x`/`bb_y`/`bb_z`. The
    # classifier itself runs upstream, on the `ObjectCentricState` (see `predicates.py`);
    # these features are what let a test -- and a reader -- see the same rest pose the
    # classifier judged, rather than having to take its verdict on trust.
    cube_type: ClassVar[Type] = Type(
        name="tossing3d_cube",
        feature_names=("x", "y", "z", "qx", "qy", "qz", "qw", "bb_x", "bb_y", "bb_z"),
    )
    # `bin_0` and `cuboid_barrier` are `MujocoObjectType`; x/y/z is what the symbolic layer
    # needs of their poses.
    #
    # The bin carries six more, and they are **not** KINDER features: they are the live
    # `Region.bbox` of `blocks_goal_region`, the box `_check_goals()` actually scores
    # against. They ride on the bin because this domain assumes the bin's interior *is*
    # that region -- see `predicates.py`'s module docstring for the assumption, the config
    # that makes it true, and the config where it is false. Carrying the box in the `State`
    # (rather than having a predicate reach for a class attribute on the Environment) is
    # what keeps every classifier a pure function of `State` while still agreeing with
    # `_check_goals()` exactly.
    bin_type: ClassVar[Type] = Type(
        name="tossing3d_bin",
        feature_names=("x", "y", "z", "x_min", "y_min", "z_min", "x_max", "y_max", "z_max"),
    )
    barrier_type: ClassVar[Type] = Type(name="tossing3d_barrier", feature_names=("x", "y", "z"))
    # Also ours, and also not a KINDER object: the two facts a flat State cannot otherwise
    # carry. `seed` is what `set_state` rebuilds the scene from; `steps_taken` is what
    # lets it refuse a state it cannot restore.
    scene_type: ClassVar[Type] = Type(name="tossing3d_scene", feature_names=("seed", "steps_taken"))

    robot: ClassVar[Object] = Object(name="robot", type=robot_type)
    cube: ClassVar[Object] = Object(name="cube_0", type=cube_type)
    bin: ClassVar[Object] = Object(name="bin_0", type=bin_type)
    barrier: ClassVar[Object] = Object(name="cuboid_barrier", type=barrier_type)
    scene: ClassVar[Object] = Object(name="scene", type=scene_type)

    # Skill ids, as they appear in slot 0 of an Action. Fixed so a recorded action vector
    # keeps its meaning.
    #
    # **These do not mean what they meant before the two-skill migration.** Id 1 was
    # `MoveToThrowPose` and id 2 was `Toss`; both are now one composed skill at id 1, and
    # id 2 is unused. An action vector recorded before that migration is not readable
    # here -- the parameter slots widened too.
    pick_cube_id: ClassVar[int] = 0
    move_to_toss_location_and_toss_id: ClassVar[int] = 1
    # Not a skill: the id `noop_action` carries, chosen outside the real ids so
    # `_execute` falls through every branch. Negative rather than 2 so that adding a
    # third controller can never silently turn every no-op into it.
    noop_id: ClassVar[int] = -1

    # [skill_id, param_0, param_1, param_2, param_3]. Four parameter slots because the
    # composed toss is the widest skill (distance, rotation, release speed, gripper
    # release ms); `PickCube` uses none at all, and the unused slots are ignored rather
    # than validated, exactly as Tossing Room does.
    action_space: ClassVar[Box] = Box(-np.inf, np.inf, (5,))

    variant: str = "o1"
    scene_bg: bool = True
    # Upstream's own `test_pick_ground_toss` seed, and the one every number in this
    # domain's docs was measured at. `hard_reset` uses it; `Tasks` supplies its own.
    canonical_seed: int = 125

    _backend: KinderBackend | None = PrivateAttr(default=None)
    _last_skill_error: str | None = PrivateAttr(default=None)
    _last_controller_steps: tuple[int, ...] = PrivateAttr(default=())

    def backend(self) -> KinderBackend:
        """The live simulator handle, built on first use.

        Built lazily rather than in `model_post_init` so that constructing a
        `Tossing3DEnvironment` -- which every offline test does -- does not import KINDER.

        **This domain runs the scene the installed KINDER ships, and no longer selects
        one.** There used to be a `Tossing3DTaskConfig` enum here, and no `task_config_path`
        is passed now because there is nothing left to choose:

        - Upstream commit `1183de7` had moved `bin_init_region` from x = 2.0 to x = 2.23
          and left `blocks_goal_region` behind, so the bin sat 23 cm past the box that
          scores and **a cube landing in the bin was a scored failure**. Training against
          that scene would have rewarded missing the bin.
        - This repo shipped `scripts/task_configs/Tossing3D-o1-coincident.json` to put the
          bin back, and the enum's `COINCIDENT` selected it against `STOCK`, which passed
          nothing and let KINDER load its own registered `Tossing3D-o1.json`.
        - The fix then landed upstream -- `kindergarden` PR #126 -- **as an edit to
          `Tossing3D-o1.json` itself rather than as a new variant**. Both enum members
          therefore came to load the same scene, and two tests asserting the contrast
          broke with nobody having edited them.

        Josh's call was to keep it that way and to take upstream's file as the file, so
        neither the enum nor this repo's copy of the scene survives.

        **The cost of that, stated rather than implicit: the scene now moves with the
        `reference/kindergarden` pin.** That coupling is exactly what made `STOCK`'s
        meaning drift, and it is accepted here rather than unnoticed --
        `test_the_shipped_scene_still_puts_the_bin_on_the_box_that_scores` pins the
        property that matters (bin centred on the goal region) directly against the JSON
        the installed KINDER ships, so a pin bump that moves the bin off the region again
        fails loudly instead of silently re-breaking the domain.

        `o2` is refused rather than silently run: its scene needs two cubes in the goal
        region and this domain's symbolic layer is single-cube, so a run labelled `o2`
        would be measuring something this package cannot describe.
        """
        if self._backend is None:
            if self.variant != "o1":
                raise ValueError(
                    f"this domain's symbolic layer describes the o1 scene, not "
                    f"{self.variant!r}; use --variant o1"
                )
            self._backend = KinderBackend(
                env_id=f"kinder/Tossing3D-{self.variant}-v0",
                scene_bg=self.scene_bg,
            )
        return self._backend

    def last_skill_error(self) -> str | None:
        """Why the most recent `take_action` was a no-op, or `None` if it was not one.

        KINDER's motion planners `assert plan is not None`, so an unreachable grasp or an
        unplannable arm trajectory raises out of a controller's `reset`. `take_action`
        must be total over its action space, so that becomes a no-op transition -- but it
        is recorded here rather than discarded, because "the sampler drew badly" and
        "upstream changed under us" look identical from the outside otherwise.
        """
        return self._last_skill_error

    def last_controller_steps(self) -> tuple[int, ...]:
        """`env.step` counts for the controller executions the last skill ran.

        One entry per upstream controller. Both skills are now a single controller each --
        the composed toss drives base motion, windup and swing through one `step()` -- so
        this is a one-tuple either way. It used to report the oracle's four executions as
        `71 / 23 / 16 / 18` in `docs/kinder-environment-validation.md`; that breakdown is
        no longer observable from here, because the phases are inside one controller.
        """
        return self._last_controller_steps

    def build_state(
        self,
        *,
        observation: KinderObservation,
        seed: int,
        steps_taken: int,
        object_centric: Any = None,
        abstract_atoms: frozenset[AbstractAtom] | None = None,
    ) -> Tossing3DState:
        """Translate one `KinderObservation` into this domain's `core.State`.

        Pure, and free of KINDER: it reads named features out of plain dicts, so a test
        can hand it a hand-built observation and check the translation with no simulator.

        `object_centric` and `abstract_atoms` are the symbolic half, and default to
        absent precisely so that purity survives -- a caller checking the *translation*
        needs no simulator, and gets a state that says so rather than one that silently
        answers `False` to every predicate. `_observed_state` is what fills them in on
        the live path; see `types.py` for why they travel on the state at all.
        """
        return Tossing3DState(
            object_centric=object_centric,
            abstract_atoms=abstract_atoms,
            data={
                self.robot: self._vector(
                    observation=observation,
                    name=self._robot_source(observation=observation),
                    obj=self.robot,
                ),
                self.cube: self._vector(
                    observation=observation, name=self.cube.name, obj=self.cube
                ),
                # Pose from KINDER, then the scored box appended: the bin is the one object
                # whose features come from two sources, because the box is not KINDER's
                # notion of the bin at all -- it is `blocks_goal_region`, which this domain
                # assumes the bin's interior coincides with.
                self.bin: np.concatenate([
                    self._vector(
                        observation=observation,
                        name=self.bin.name,
                        obj=self.bin,
                        features=("x", "y", "z"),
                    ),
                    np.array(observation.goal_region, dtype=float),
                ]),
                self.barrier: self._vector(
                    observation=observation, name=self.barrier.name, obj=self.barrier
                ),
                self.scene: np.array([float(seed), float(steps_taken)], dtype=float),
            },
        )

    def _observed_state(self, *, seed: int, steps_taken: int) -> Tossing3DState:
        """The live simulator's current state, translated *and* abstracted.

        One place, so the flat features and the symbolic atoms are always read from the
        same instant. `abstract_atoms` runs upstream's own classifiers against the
        KINDER state being captured here, which is what makes the resulting `State`
        self-contained: it answers its own predicates later without consulting the
        simulator again.
        """
        backend = self.backend()
        return self.build_state(
            observation=backend.observe(),
            seed=seed,
            steps_taken=steps_taken,
            object_centric=backend.snapshot(),
            abstract_atoms=backend.abstract_atoms(),
        )

    def take_action(self, *, action: Action) -> State:
        """Run one whole skill in the live simulator and return the resulting state.

        Total over the action space, like every other domain here: an unrecognised skill
        id, a non-finite parameter, or a controller whose motion planning fails is a
        no-op that still advances `steps_taken`. A skill that *ran* and simply did not
        achieve what it wanted is not a no-op -- the world moved -- and is not
        distinguished here; that is competence, and the harness measures it by whether
        the goal ends up satisfied.
        """
        state = self.get_current_state()
        seed = int(round(state.get(obj=self.scene, feature_name="seed")))
        steps_taken = int(round(state.get(obj=self.scene, feature_name="steps_taken")))
        self._last_skill_error = None
        self._last_controller_steps = ()

        runs = self._execute(action=action)
        self._last_controller_steps = tuple(run.steps for run in runs)
        errors = [run.error for run in runs if run.error is not None]
        if errors:
            self._last_skill_error = "; ".join(errors)

        next_state = self._observed_state(seed=seed, steps_taken=steps_taken + 1)
        # `_adopt`, deliberately not `set_state`: the simulator has already advanced by
        # this skill, so there is nothing to restore, and `set_state` would refuse a
        # `steps_taken > 0` state anyway. `set_state`'s job is the privileged *external*
        # override; this is the domain's own forward dynamics.
        self._adopt(state=next_state)
        return next_state

    def get_valid_actions(self) -> list[Action]:
        """Empty, as in every continuous-parameter domain here: the parameter slots range
        over the reals, so there is no finite enumeration to return. A `Method` picks a
        `GroundSkill` and samples its parameters instead (see `skill_provider.py`)."""
        return []

    def noop_action(self) -> Action:
        """`noop_id` in slot 0, which `_execute` falls through as an unrecognised skill.

        Emphatically not a zero vector: `pick_cube_id == 0`, so `np.zeros(5)` is a real
        `pick_cube` -- a whole arm trajectory, and the concrete bug this method exists to
        close. This is the one domain here where a wrong no-op costs seconds of simulator
        time as well as a wrong state.

        `take_action` still advances the scene's `steps_taken`, as it does for any
        unrecognised action. That is the interface's contract, not a violation of it:
        the world does not move, but the transition is still charged.
        """
        return np.array([float(self.noop_id), 0.0, 0.0, 0.0, 0.0])

    def set_state(self, *, state: State) -> None:
        """Adopt `state`, rebuilding the simulator from its seed when it is an initial one.

        Two cases, and the distinction is the point:

        - `steps_taken == 0`: rebuild the scene at the state's own seed and adopt the
          freshly observed result. That is a genuine rewind, and it is what
          `Problem.reset_to_task` and `Method.reset_environment` need.
        - `steps_taken > 0`: **raise**. A mid-episode MuJoCo state is not recoverable from
          a flat feature vector, and quietly restoring the episode's *initial* state
          instead would make an evaluation look like it rewound when it did not.

        The state adopted is the one freshly observed from the rebuilt scene, not the
        argument -- they agree, because the same seed produces the same scene, and taking
        the observed one means what this environment reports is always what the simulator
        actually is.

        `take_action` deliberately does **not** come through here: advancing the world by
        a skill is this domain's forward dynamics, while `set_state` is the privileged
        external override (see `core.Environment`).
        """
        steps_taken = int(round(state.get(obj=self.scene, feature_name="steps_taken")))
        if steps_taken != 0:
            raise ValueError(
                "Tossing3DEnvironment.set_state can only restore an episode-initial "
                f"state, but this one has steps_taken={steps_taken}. A flat core.State "
                "cannot carry MuJoCo's qpos/qvel, so there is no faithful mid-episode "
                "rewind in this domain -- see the class docstring."
            )
        self.reset_to_seed(seed=int(round(state.get(obj=self.scene, feature_name="seed"))))

    def hard_reset(self) -> None:
        """Rebuild the scene at `canonical_seed`. Harness-only, before a run starts."""
        self.reset_to_seed(seed=self.canonical_seed)

    def reset_to_seed(self, *, seed: int) -> State:
        """Rebuild the scene at `seed`, adopt the resulting initial `State`, and return it.

        Public because `Tasks` needs it: the only way to obtain an initial state in this
        domain is to actually build the scene, so task sampling is a simulator operation
        rather than an arithmetic one -- and therefore one with a side effect. Sampling a
        task leaves the live simulator sitting at that task, which is why `Tasks`' own
        docstring warns callers that drawing several tasks up front gets them the last
        one's scene.

        Adopting rather than only returning is what makes `take_action` legal
        immediately afterwards; `hard_reset` and `set_state` both go through here, so
        there is exactly one place that puts this domain into a known state.
        """
        self.backend().reset(seed=seed)
        state = self._observed_state(seed=seed, steps_taken=0)
        self._adopt(state=state)
        return state

    def reset_movables(self) -> bool:
        """Reposition `cube`/`bin` to freshly sampled ground poses, leaving the
        robot's own live configuration untouched, and return True: unlike the
        `core.Environment` default (which declines), Tossing3D genuinely supports
        this, backed by a real per-object pose-setting primitive in the live
        simulator -- see `KinderBackend.reset_cube_and_bin`.

        **Deliberately not routed through `set_state`.** `set_state` can only rebuild
        the *whole* scene from a seed (see its own docstring), which would relocate
        the robot too -- exactly what this method exists to avoid. This is a third,
        separate operation, alongside `set_state` and `snapshot`/`restore`, for the
        same reason those two stayed separate from each other: each is a genuinely
        different kind of state change, and widening one to cover another would
        misdescribe what actually happened.

        **`steps_taken`/`seed` carry forward unchanged.** Unlike `take_action` (which
        advances `steps_taken` by one, as this domain's own forward dynamics) and
        `set_state` (which always reports `steps_taken=0`, since it rebuilds the scene
        from `seed`), this is neither: a live-simulator partial correction is not a
        skill execution and is not a scene rebuild, so nothing about this domain's own
        episode bookkeeping moves. Carrying them forward is also what keeps a second
        `reset_movables` mid-episode from looking like `set_state` accepted a non-zero
        `steps_taken` state, which it explicitly refuses to do.
        """
        state = self.get_current_state()
        seed = int(round(state.get(obj=self.scene, feature_name="seed")))
        steps_taken = int(round(state.get(obj=self.scene, feature_name="steps_taken")))
        backend = self.backend()
        backend.reset_cube_and_bin()
        next_state = self.build_state(
            observation=backend.observe(),
            seed=seed,
            steps_taken=steps_taken,
            object_centric=backend.snapshot(),
            abstract_atoms=backend.abstract_atoms(),
        )
        self._adopt(state=next_state)
        return True

    def snapshot(self) -> "Tossing3DSnapshot":
        """A restorable handle to the live simulator, including mid-episode.

        `set_state` cannot do this from a `core.State` alone, because a `core.State` is a
        lossy projection of KINDER's own state (four of the robot's twenty-two features,
        six of the cube's sixteen). KINDER's `ObjectCentricState` is *not* lossy -- it
        carries velocities and the full arm configuration, which is exactly why upstream's
        `tidybot3d_shelf3D.py` can use it to build a transition function -- so a genuine
        rewind is available as long as the caller holds one of these.

        Deliberately a separate, explicitly-named operation rather than a widening of
        `set_state`: `core.Environment.set_state` is documented as the *human's*
        privileged override, and speculatively executing a skill and rewinding is not
        that. It is what a fidelity check does, and
        `tests/environments/tossing3d/test_operator_fidelity.py` is its one caller.
        """
        return Tossing3DSnapshot(
            kinder_state=self.backend().snapshot(),
            state=self.get_current_state().model_copy(deep=True),
        )

    def restore(self, *, snapshot: "Tossing3DSnapshot") -> State:
        """Put the simulator and this environment back to `snapshot`, exactly."""
        self.backend().restore(snapshot=snapshot.kinder_state)
        self._adopt(state=snapshot.state.model_copy(deep=True))
        return self.get_current_state()

    def is_solved(self) -> bool:
        """Upstream's own `_check_goals()`, straight through.

        Used by the fidelity tests to check `predicates.IN_BIN` against the thing
        it is supposed to agree with. Not used to decide episode success -- that goes
        through `Goal.is_satisfied` like every other domain, so the symbolic layer is
        what is actually being trusted.
        """
        return self.backend().check_goals()

    def close(self) -> None:
        """Release the simulator. Idempotent, and safe before any reset."""
        if self._backend is not None:
            self._backend.close()

    def _execute(self, *, action: Action) -> list[ControllerRun]:
        """Dispatch one action vector onto upstream's controllers."""
        backend = self.backend()
        if not np.all(np.isfinite(np.asarray(action, dtype=float))):
            self._last_skill_error = f"non-finite action: {action!r}"
            return []
        skill_id = int(round(float(action[0])))
        if skill_id == self.pick_cube_id:
            # No parameters at all: upstream derives the standoff and the grasp rotation
            # internally, so every slot is ignored here rather than read and discarded.
            return [backend.run_pick_cube()]
        if skill_id == self.move_to_toss_location_and_toss_id:
            return [
                backend.run_move_to_toss_location_and_toss(
                    distance=float(action[1]),
                    rotation=float(action[2]),
                    release_speed_deg_s=float(action[3]),
                    gripper_release_ms=float(action[4]),
                )
            ]
        self._last_skill_error = f"unknown skill id: {skill_id}"
        return []

    def _adopt(self, *, state: State) -> None:
        self.current_state = state

    def _robot_source(self, *, observation: KinderObservation) -> str:
        """The robot's name in the observation.

        Resolved by feature schema rather than by literal: the robot is the one object
        whose name comes from the robot config rather than the task JSON's `objects`
        block, and it is the only one carrying `pos_base_x`.
        """
        for name, features in observation.features.items():
            if "pos_base_x" in features:
                return name
        raise KeyError(
            "no object in this observation has a pos_base_x feature, so none of them is "
            f"the TidyBot robot: {sorted(observation.features)}"
        )

    def _vector(
        self,
        *,
        observation: KinderObservation,
        name: str,
        obj: Object,
        features: tuple[str, ...] | None = None,
    ) -> np.ndarray:
        """The named features of one observed object, in schema order.

        `features` narrows the read to a prefix of the object's schema, for the one object
        (the bin) whose remaining features do not come from KINDER at all.
        """
        return np.array(
            [
                observation.get(name=name, feature=feature)
                for feature in (features if features is not None else obj.type.feature_names)
            ],
            dtype=float,
        )
