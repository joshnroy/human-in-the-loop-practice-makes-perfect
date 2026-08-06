"""The guard that makes the identity-vs-causal comparison trustworthy.

`tossingroomsplitidentity` is a verbatim fork of `tossingroomsplit` with **exactly one
delta**: the throw representation. The identity domain puts `target_force` on the item
and tests `|force - item.target_force| < throw_tolerance`; the causal domain puts a
`throw_distance` on the bin and a `weight` on the item and tests against an unobserved
affine `required_force(distance, weight)`.

Everything else must agree, because the experiment lays the two arms side by side and
reads the difference as "representation". A fork that drifts silently -- and
`tossingroomsplit` is itself a verbatim fork of `tossingroom`, with nothing propagating
automatically in either direction -- would invalidate the whole comparison. So the
agreement is asserted here rather than left to a reviewer's diff.

The permitted differences are enumerated in ONE place (`PERMITTED_TYPE_DELTAS` below)
and every other structural fact is required to match. Adding a new difference means
editing that set, which is exactly the moment a reviewer should be asked whether the
comparison still means what it claims.
"""

import numpy as np
import pytest

from hitl_pmp.core.method.types import GroundSkill
from hitl_pmp.environments.tossingroomsplit.environment import TossingRoomSplitEnvironment
from hitl_pmp.environments.tossingroomsplit.problem import TossingRoomSplitProblem
from hitl_pmp.environments.tossingroomsplit.skill_provider import TossingRoomSplitSkillProvider
from hitl_pmp.environments.tossingroomsplit.tasks import TossingRoomSplitTasks
from hitl_pmp.environments.tossingroomsplitidentity.environment import (
    TossingRoomSplitIdentityEnvironment,
)
from hitl_pmp.environments.tossingroomsplitidentity.problem import (
    TossingRoomSplitIdentityProblem,
)
from hitl_pmp.environments.tossingroomsplitidentity.skill_provider import (
    TossingRoomSplitIdentitySkillProvider,
)
from hitl_pmp.environments.tossingroomsplitidentity.skills import (
    TossingRoomSplitIdentitySkills,
)
from hitl_pmp.environments.tossingroomsplitidentity.tasks import (
    TossingRoomSplitIdentityGoalType,
    TossingRoomSplitIdentityTasks,
)
from hitl_pmp.methods.practice_makes_perfect.ees_method import EesMethod

# The item and bin types are the ONLY types allowed to differ, and only in their feature
# schemas -- that IS the throw representation. Every other type must be feature-identical.
PERMITTED_TYPE_DELTAS = frozenset({"trash", "recycling", "trash_bin", "recycling_bin"})


@pytest.fixture
def causal_env() -> TossingRoomSplitEnvironment:
    return TossingRoomSplitEnvironment()


@pytest.fixture
def identity_env() -> TossingRoomSplitIdentityEnvironment:
    return TossingRoomSplitIdentityEnvironment()


def test_room_layout_and_ledge_direction_agree(*, causal_env, identity_env):
    """Same hallway, same bins, same one-way ledge in the same direction."""
    for field in (
        "num_rooms",
        "start_room",
        "recycling_bin_room",
        "trash_bin_room",
        "blocked_right_from",
        "throw_tolerance",
    ):
        assert getattr(identity_env, field) == getattr(causal_env, field), field

    # The ledge is a *behaviour*, not just a number: the rightward step out of
    # blocked_right_from must be refused and the leftward step back allowed, identically.
    for env in (causal_env, identity_env):
        env.hard_reset()
        ledge = env.blocked_right_from
        env.set_state(state=env.get_current_state().model_copy(deep=True))
        state = env.get_current_state()
        state.set(obj=env.robot, feature_name="room", feature_val=float(ledge))
        env.set_state(state=state)
        env.take_action(action=np.array([float(env.SKILL_MOVE_ROOM), float(ledge + 1), 0.0]))
        blocked_room = int(round(env.get_current_state().get(obj=env.robot, feature_name="room")))
        assert blocked_room == ledge, "rightward step across the ledge must be refused"

        env.take_action(action=np.array([float(env.SKILL_MOVE_ROOM), float(ledge - 1), 0.0]))
        back_room = int(round(env.get_current_state().get(obj=env.robot, feature_name="room")))
        assert back_room == ledge - 1, "leftward step must be allowed"


def test_bin_capacity_and_button_wiring_agree(*, causal_env, identity_env):
    """Capacity-1 bins, and each button sitting in (and emptying) its own bin's room."""
    assert identity_env.BIN_CAPACITY == causal_env.BIN_CAPACITY == 1

    for kind_name in ("TRASH_KIND", "RECYCLING_KIND"):
        causal_kind = getattr(causal_env, kind_name)
        identity_kind = getattr(identity_env, kind_name)
        assert identity_kind == causal_kind, kind_name
        assert identity_env.bin_room_for_kind(kind=identity_kind) == causal_env.bin_room_for_kind(
            kind=causal_kind
        )
        assert identity_env.button_room_for_kind(
            kind=identity_kind
        ) == causal_env.button_room_for_kind(kind=causal_kind)
        assert (
            identity_env.button_for_kind(kind=identity_kind).name
            == causal_env.button_for_kind(kind=causal_kind).name
        )
        assert (
            identity_env.bin_for_kind(kind=identity_kind).name
            == causal_env.bin_for_kind(kind=causal_kind).name
        )


def test_lifted_skill_names_and_arities_agree(*, causal_env, identity_env):
    """Same seven lifted skills, same parameter arity, same continuous param_dim, and
    the same symbolic preconditions/effects by predicate name. `param_dim` is what
    decides which skills get a `LearnedSkillSampler` at all, so a drift here would
    change the experiment outright."""
    causal = {skill.name: skill for skill in TossingRoomSplitSkillProvider(env=causal_env).skills()}
    identity = {
        skill.name: skill
        for skill in TossingRoomSplitIdentitySkillProvider(env=identity_env).skills()
    }
    assert set(identity) == set(causal)

    for name, causal_skill in causal.items():
        identity_skill = identity[name]
        assert len(identity_skill.parameters) == len(causal_skill.parameters), name
        assert identity_skill.param_dim == causal_skill.param_dim, name
        # Parameter *names* in order, so the object binding order compute_action unpacks
        # is the same. The parameter TYPES are allowed to carry different schemas.
        assert [v.name for v in identity_skill.parameters] == [
            v.name for v in causal_skill.parameters
        ], name
        for attribute in ("preconditions", "add_effects", "delete_effects"):
            assert {
                (atom.predicate.name, tuple(v.name for v in atom.variables))
                for atom in getattr(identity_skill, attribute)
            } == {
                (atom.predicate.name, tuple(v.name for v in atom.variables))
                for atom in getattr(causal_skill, attribute)
            }, f"{name}.{attribute}"


def test_only_the_item_and_bin_type_schemas_differ(*, causal_env, identity_env):
    """The permitted delta, stated positively: every type but the items and bins is
    feature-identical, and those four differ exactly as the representation demands."""
    causal = {t.name: t for t in TossingRoomSplitSkillProvider(env=causal_env).types()}
    identity = {t.name: t for t in TossingRoomSplitIdentitySkillProvider(env=identity_env).types()}
    assert set(identity) == set(causal)

    differing = {
        name
        for name, causal_type in causal.items()
        if identity[name].feature_names != causal_type.feature_names
    }
    assert differing == PERMITTED_TYPE_DELTAS

    # And the delta is the throw representation specifically, not some other drift.
    for item in ("trash", "recycling"):
        assert causal[item].feature_names == ("kind", "weight")
        assert identity[item].feature_names == ("kind", "target_force")
    for bin_name in ("trash_bin", "recycling_bin"):
        assert causal[bin_name].feature_names == ("count", "room", "kind", "throw_distance")
        assert identity[bin_name].feature_names == ("count", "room", "kind")


def test_predicate_names_agree(*, causal_env, identity_env):
    assert {
        p.name for p in TossingRoomSplitIdentitySkillProvider(env=identity_env).predicates()
    } == {p.name for p in TossingRoomSplitSkillProvider(env=causal_env).predicates()}


@pytest.mark.parametrize("num_test_tasks", [10, 30])
def test_test_task_composition_agrees(*, causal_env, identity_env, num_test_tasks):
    """14 TRASH / 14 RECYCLING / 2 EMPTY at 30 tasks, in both arms."""
    causal = TossingRoomSplitTasks(env=causal_env, num_test_tasks=num_test_tasks)
    identity = TossingRoomSplitIdentityTasks(env=identity_env, num_test_tasks=num_test_tasks)
    causal_counts = {
        goal_type.value: count for goal_type, count in causal.test_goal_type_counts().items()
    }
    identity_counts = {
        goal_type.value: count for goal_type, count in identity.test_goal_type_counts().items()
    }
    assert identity_counts == causal_counts
    if num_test_tasks == 30:
        assert identity_counts == {"trash": 14, "recycling": 14, "empty": 2}


def test_horizon_agrees(*, causal_env, identity_env):
    """Horizon 12 on the default layout, in both arms."""
    causal = TossingRoomSplitProblem(
        env=causal_env, tasks=TossingRoomSplitTasks(env=causal_env, num_test_tasks=30)
    )
    identity = TossingRoomSplitIdentityProblem(
        env=identity_env,
        tasks=TossingRoomSplitIdentityTasks(env=identity_env, num_test_tasks=30),
    )
    assert identity.max_episode_steps() == causal.max_episode_steps() == 12
    assert identity.longest_shortest_solve() == causal.longest_shortest_solve()


@pytest.mark.parametrize("seed", [0, 1, 7])
def test_the_two_arms_draw_the_same_tasks(*, causal_env, identity_env, seed):
    """**The strongest form of matched difficulty**, and what makes the arms *paired*
    rather than merely comparable.

    Both `Tasks` draw the same two causes, from the same ranges, in the same order, and
    resolve them with the same five constants. So they consume their RNG in lockstep and,
    at any seed, present the identical sequence of goal families with the identical
    required force for every throw. The only difference is where that force ends up: the
    causal arm puts the two causes in the State and keeps the force out, this arm puts
    the force in and drops the causes.

    Asserted over the TRAIN stream (whose goal families are sampled, so a divergence in
    RNG consumption shows up immediately as a family mismatch) and over both throw
    families' targets."""
    causal = TossingRoomSplitTasks(env=causal_env, seed=seed, num_test_tasks=30)
    identity = TossingRoomSplitIdentityTasks(env=identity_env, seed=seed, num_test_tasks=30)

    for _ in range(40):
        causal_task = causal.sample_train_task()
        identity_task = identity.sample_train_task()
        assert {atom.predicate.name for atom in causal_task.goal.atoms} == {
            atom.predicate.name for atom in identity_task.goal.atoms
        }, "the two arms' goal-family sequences diverged, so their RNG is not in lockstep"

        for causal_item, causal_bin, identity_item in (
            (causal_env.trash, causal_env.trash_bin, identity_env.trash),
            (causal_env.recycling, causal_env.recycling_bin, identity_env.recycling),
        ):
            causal_required = causal_env.required_force(
                throw_distance=float(
                    causal_task.initial_state.get(obj=causal_bin, feature_name="throw_distance")
                ),
                item_weight=float(
                    causal_task.initial_state.get(obj=causal_item, feature_name="weight")
                ),
            )
            identity_target = float(
                identity_task.initial_state.get(obj=identity_item, feature_name="target_force")
            )
            assert identity_target == pytest.approx(causal_required, abs=1e-12), (
                "the same task requires a different force in the two arms"
            )


def test_random_force_lands_with_matched_probability(*, causal_env, identity_env):
    """A uniformly random force (the U(0, 1) band `sample_params` draws from) lands with
    probability 0.20 on EVERY task, in both arms -- no task's winning window
    `[required - tolerance, required + tolerance]` is clipped by the edge of that band.

    Asserted analytically over sampled tasks rather than by throwing: the quantity that
    matters is how much of the window survives intersection with (0, 1)."""
    causal = TossingRoomSplitTasks(env=causal_env, seed=0, num_test_tasks=30)
    identity = TossingRoomSplitIdentityTasks(env=identity_env, seed=0, num_test_tasks=30)
    tolerance = identity_env.throw_tolerance

    required: list[float] = []
    for _ in range(200):
        causal.sample_train_task()
        task = identity.sample_train_task()
        for item in (identity_env.trash, identity_env.recycling):
            required.append(float(task.initial_state.get(obj=item, feature_name="target_force")))
    windows = np.asarray(required)
    covered = np.minimum(windows + tolerance, 1.0) - np.maximum(windows - tolerance, 0.0)
    assert np.allclose(covered, 2 * tolerance), (
        "a task's winning window is clipped by the [0, 1) force band, so a uniformly "
        "random force does not land with probability 0.20 on every task"
    )


def test_the_best_state_blind_force_is_matched_too(*, causal_env, identity_env):
    """The second difficulty axis, and the one a plain `Uniform` target would have got
    wrong -- so it is pinned rather than assumed.

    A sampler that ignores the state entirely can still pick one fixed force. How well
    that does depends on the *marginal* distribution of the required force, not just its
    span. The causal arm's is triangular (a sum of two uniforms), concentrating mass near
    0.5; a `Uniform[0.1, 0.9)` target would have been flat, and the best fixed force
    would have landed roughly 119/400 against the causal arm's 185/400. Resolving the
    same two causes reproduces the marginal exactly, so the two ceilings coincide and a
    cross-arm reading of "what did conditioning on the state buy" is not confounded."""
    causal = TossingRoomSplitTasks(env=causal_env, seed=3, num_test_tasks=30)
    identity = TossingRoomSplitIdentityTasks(env=identity_env, seed=3, num_test_tasks=30)
    tolerance = identity_env.throw_tolerance

    causal_required: list[float] = []
    identity_required: list[float] = []
    for _ in range(400):
        causal_task = causal.sample_train_task()
        identity_task = identity.sample_train_task()
        causal_required.append(
            causal_env.required_force(
                throw_distance=float(
                    causal_task.initial_state.get(
                        obj=causal_env.trash_bin, feature_name="throw_distance"
                    )
                ),
                item_weight=float(
                    causal_task.initial_state.get(obj=causal_env.trash, feature_name="weight")
                ),
            )
        )
        identity_required.append(
            float(
                identity_task.initial_state.get(obj=identity_env.trash, feature_name="target_force")
            )
        )

    def best_fixed_force_hits(*, required: list[float]) -> int:
        values = np.asarray(required)
        candidates = np.linspace(0.0, 1.0, 501)
        return int(
            np.max(np.sum(np.abs(values[None, :] - candidates[:, None]) < tolerance, axis=1))
        )

    assert best_fixed_force_hits(required=identity_required) == best_fixed_force_hits(
        required=causal_required
    )


def test_target_force_sits_at_input_index_four(*, identity_env):
    """The point of this arm, pinned: in a throw's classifier row
    `[1.0] + concat(state[obj] for obj in ground_skill.objects) + [force]`, the answer is
    a literal column, and the optimal policy is "copy input index 4".

    The row is built through the REAL pipeline -- a `GroundSkill` over
    `THROW_TRASH.parameters`, then `EesMethod.sampler_input_row` -- rather than from a
    hand-written object tuple. A test that lists the objects itself only asserts a
    property of its own list: reorder `THROW_TRASH.parameters` and the real row moves
    while the test stays green, which is precisely the failure this arm cannot afford,
    since "index 4" is the whole claim."""
    tasks = TossingRoomSplitIdentityTasks(env=identity_env, num_test_tasks=30)
    task = tasks.build_task(
        goal_type=TossingRoomSplitIdentityGoalType.TRASH, rng=np.random.default_rng(0)
    )
    throw = TossingRoomSplitIdentitySkills.THROW_TRASH
    # Bind each lifted parameter to the one object of its type, in the skill's own order.
    by_type = {
        obj.type: obj
        for obj in TossingRoomSplitIdentitySkillProvider(env=identity_env).objects()
        if obj.type
        in {identity_env.robot_type, identity_env.trash_type, identity_env.trash_bin_type}
    }
    rooms = identity_env.get_rooms()
    objects = tuple(
        by_type.get(variable.type, rooms[identity_env.trash_bin_room])
        for variable in throw.parameters
    )
    method = EesMethod(
        env=identity_env,
        skill_provider=TossingRoomSplitIdentitySkillProvider(env=identity_env),
        seed=0,
    )
    row = method.sampler_input_row(
        ground_skill=GroundSkill(skill=throw, objects=objects),
        state=task.initial_state,
        params=np.array([0.42]),
    )
    expected = float(task.initial_state.get(obj=identity_env.trash, feature_name="target_force"))
    assert len(row) == 11, "the identity arm's throw row is 11 columns, not the causal 12"
    assert row[4] == expected, "target_force is not at index 4"
    assert row[10] == 0.42, "the sampled force is not the last column"


def test_the_two_way_ledge_flag_is_a_causal_only_divergence_that_defaults_off(*, causal_env):
    """`--two-way-ledge` (the reset-free positive control) exists on the causal domain
    only -- the identity fork has no such field. That is a real divergence, so it is
    recorded here rather than left for a reviewer to notice.

    It is harmless to this guard for exactly one reason: it defaults OFF, and every
    assertion in this file constructs both environments at their defaults. So the two
    worlds compared above are still the same world. If the default ever flips, or if the
    identity fork grows its own copy, this test is the thing that has to be revisited --
    and the throw-representation comparison re-baselined."""
    assert causal_env.two_way_ledge is False
    assert "two_way_ledge" not in TossingRoomSplitIdentityEnvironment.model_fields
    assert causal_env.ledge_blocks_rightward(from_room=causal_env.blocked_right_from) is True
