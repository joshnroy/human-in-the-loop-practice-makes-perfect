"""The per-skill trace collector, and the one property that licenses using it.

`stats.json` records tasks solved. The question this domain poses is per *skill* --
how many times was `ThrowTrash` practiced against `ThrowRecycling`, how often did each
succeed, and where did each one's competence get to -- and none of that leaves
`EesMethod`'s internals. So `scripts/tossingroomsplit_skill_traces.py` subclasses the
real method to read it out, the same way `scripts/tossingroom_throw_traces.py` already
does for the throw force.

**The load-bearing test here is `test_tracing_does_not_perturb_the_run`.** The
experiment reports the sweep's `stats.json` success curves and this collector's
per-skill counts side by side and treats them as one experiment measured twice. That is
only honest if instrumenting the run changes nothing about it -- so the traced run's
per-sweep `(transitions, solved, total)` triples must equal an untraced run's, exactly,
at the same seed.
"""

import numpy as np

from hitl_pmp.core.method.types import GroundSkill
from hitl_pmp.core.metrics.metrics import Metrics
from hitl_pmp.environments.tossingroomsplit.environment import TossingRoomSplitEnvironment
from hitl_pmp.environments.tossingroomsplit.problem import TossingRoomSplitProblem
from hitl_pmp.environments.tossingroomsplit.skill_provider import TossingRoomSplitSkillProvider
from hitl_pmp.environments.tossingroomsplit.skills import TossingRoomSplitSkills
from hitl_pmp.environments.tossingroomsplit.tasks import TossingRoomSplitTasks
from hitl_pmp.methods.practice_makes_perfect.ees_method import EesMethod
from hitl_pmp.practice_loop import PracticeLoop
from scripts.tossingroomsplit_skill_traces import (
    PeriodLog,
    SkillTraceCollector,
    TracingEesMethod,
)

# Small but real: 3 cycles is enough for the collector to have drained twice and for
# both throws to have been reached, and cheap enough to keep the suite fast.
_CYCLES = 3
_STEPS = 60
_TEST_TASKS = 6
_ITERS = 100


def _traced() -> dict:
    return SkillTraceCollector.run_seed(
        seed=0,
        sampler_iters=_ITERS,
        num_cycles=_CYCLES,
        max_steps=_STEPS,
        num_test_tasks=_TEST_TASKS,
    )


def _untraced_evaluations() -> list[tuple[int, int, int]]:
    """The identical run through the stock classes -- no subclass, no logging."""
    env = TossingRoomSplitEnvironment()
    tasks = TossingRoomSplitTasks(env=env, seed=0, num_test_tasks=_TEST_TASKS)
    problem = TossingRoomSplitProblem(env=env, tasks=tasks)
    method = EesMethod(
        env=env,
        skill_provider=TossingRoomSplitSkillProvider(env=env),
        seed=0,
        sampler_max_train_iters=_ITERS,
    )
    metrics = Metrics()
    PracticeLoop.run(
        problem=problem,
        method=method,
        metrics=metrics,
        num_cycles=_CYCLES,
        max_steps_per_interaction=_STEPS,
        num_test_tasks=_TEST_TASKS,
    )
    return metrics.evaluations


def test_tracing_does_not_perturb_the_run() -> None:
    """Instrumenting must not change what happened, or the sweep's `stats.json` and this
    collector's counts are two different experiments rather than one."""
    traced = _traced()
    measured = [
        (sweep["transitions"], sweep["solved"], sweep["total"]) for sweep in traced["sweeps"]
    ]
    assert measured == _untraced_evaluations()


def test_the_two_throws_are_counted_separately() -> None:
    trace = _traced()
    practice = trace["periods"]
    names = {name for period in practice for name in period["skills"]}
    assert "ThrowTrash" in names
    assert "ThrowRecycling" in names
    # Non-vacuity: a trace where neither throw was ever practiced would satisfy any
    # claim about how their counts compare.
    assert sum(period["skills"].get("ThrowTrash", {}).get("attempts", 0) for period in practice) > 0
    assert (
        sum(period["skills"].get("ThrowRecycling", {}).get("attempts", 0) for period in practice)
        > 0
    )


def test_successes_never_exceed_attempts_for_any_skill() -> None:
    for period in _traced()["periods"]:
        for name, record in period["skills"].items():
            assert 0 <= record["successes"] <= record["attempts"], name


class TestAScoredSuccessIsNotTheSameThingAsALanding:
    """A throw's `add_effects` are `{<Kind>InBin(item, bin), HandEmpty(robot)}`, and
    `<Kind>InBin` is `count >= 1`. Before the capacity-1 redesign `HandEmpty` always held
    after a throw (a throw always released the item, hit or miss), so a throw made when
    the bin was ALREADY non-empty was scored a success **at any force at all** -- and
    asymmetrically, since the trash bin was refillable within a period and the recycling
    bin, behind the one-way ledge, was not.

    **That channel is closed on the current domain**: a bin holds at most one item, each
    throw carries its bin's empty precondition, and `_apply_throw` REFUSES a throw at a
    full bin instead of swallowing the item. The traces still record `landed` and
    `prefilled` separately from what EES scored, for two reasons: the committed
    2026-08-05 run predates the redesign and its numbers only make sense read that way,
    and `landed` re-implements the dynamics' own condition, so it has to include the
    capacity refusal or it would silently report a landing for a throw that never
    released anything.
    """

    @staticmethod
    def test_every_throw_attempt_records_whether_it_landed_and_whether_the_bin_was_prefilled() -> (
        None
    ):
        for period in _traced()["periods"]:
            for name, record in period["skills"].items():
                if not name.startswith("Throw"):
                    continue
                assert 0 <= record["landed"] <= record["attempts"], name
                assert 0 <= record["prefilled"] <= record["attempts"], name

    @staticmethod
    def test_a_landing_is_recorded_only_for_a_throw_that_really_changed_the_bin() -> None:
        """The decisive property, checked against the environment rather than against the
        add-effect check: on a run where recycling can never re-throw into its own filled
        bin, its landings and its scored successes must agree exactly. Any divergence
        means `landed` is measuring the add effects again rather than the dynamics."""
        trace = _traced()
        landed = sum(
            period["skills"].get("ThrowRecycling", {}).get("landed", 0)
            for period in trace["periods"]
        )
        prefilled = sum(
            period["skills"].get("ThrowRecycling", {}).get("prefilled", 0)
            for period in trace["periods"]
        )
        successes = sum(
            period["skills"].get("ThrowRecycling", {}).get("successes", 0)
            for period in trace["periods"]
        )
        assert prefilled == 0, "recycling cannot reach an already-filled bin in this layout"
        assert landed == successes

    @staticmethod
    def test_a_throw_refused_by_a_full_bin_is_never_recorded_as_a_landing() -> None:
        """Non-vacuity for the test above, and the one place `landed` could still go
        silently wrong after the capacity-1 redesign.

        Constructed rather than sampled, and now UNREACHABLE through a real run -- each
        throw's bin-empty precondition means EES cannot select a throw at a full bin at
        all. That is exactly why it is pinned here: `_observe_throw` re-implements the
        dynamics' landing condition, so if it dropped the capacity term it would report a
        landing for a throw the environment refused, and no end-to-end test would notice.

        The force is set to the target, so ONLY the capacity guard can refuse it. And the
        old defect is checked to be gone in the same breath: the add effects no longer
        hold, because the refused throw leaves the item in hand."""
        env = TossingRoomSplitEnvironment()
        log = PeriodLog()
        method = TracingEesMethod(
            env=env,
            skill_provider=TossingRoomSplitSkillProvider(env=env),
            seed=0,
            sampler_max_train_iters=_ITERS,
            log=log,
        )
        method.practicing = True
        state = env.build_initial_state(
            trash_target_force=0.9, recycling_target_force=0.9, trash_count=1
        )
        state.set(obj=env.robot, feature_name="room", feature_val=float(env.trash_bin_room))
        state.set(obj=env.robot, feature_name="holding", feature_val=float(env.TRASH_KIND))
        env.set_state(state=state)

        ground_skill = GroundSkill(
            skill=TossingRoomSplitSkills.THROW_TRASH,
            objects=(env.robot, env.trash, env.trash_bin, env.get_rooms()[env.trash_bin_room]),
        )
        # A force of exactly the target: a perfect throw, refused only by the full bin.
        method.pending_throws["ThrowTrash"] = method._observe_throw(
            name="ThrowTrash", state=state, force=0.9
        )
        env.take_action(
            action=TossingRoomSplitSkills.compute_action(
                ground_skill=ground_skill, params=np.array([0.9]), state=state
            )
        )
        # The old vacuous-success channel: the add effects used to hold here regardless of
        # the force. They do not any more -- the refusal leaves the item in hand, so
        # `HandEmpty` is false.
        true_atoms = method.abstract_state(state=env.get_current_state())
        assert not ground_skill.add_effects <= true_atoms

        method.observe_outcome(ground_skill=ground_skill, success=False)
        tally = log.skills["ThrowTrash"]
        assert tally.attempts == 1
        assert tally.successes == 0
        assert tally.landed == 0  # refused, so nothing landed however good the force
        assert tally.prefilled == 1  # and the full bin is why


def test_every_sweep_carries_a_per_goal_family_breakdown_that_sums_to_the_total() -> None:
    """The per-family counts are read back as `x/y`, never as a percentage, so the
    denominators have to be present and consistent with the aggregate."""
    for sweep in _traced()["sweeps"]:
        families = sweep["families"]
        assert sum(count for _solved, count in families.values()) == sweep["total"]
        assert sum(solved for solved, _count in families.values()) == sweep["solved"]


def test_competence_is_recorded_per_skill_and_stays_a_probability() -> None:
    trace = _traced()
    for cycle in trace["competence"]:
        for name, record in cycle.items():
            assert 0.0 <= record["competence"] <= 1.0, name
            # How many ground skills that lifted name was ever executed with, so a mean
            # over groundings is never quietly reported as if it were a single number.
            assert record["num_groundings"] >= 1


def test_collect_runs_exactly_the_seeds_it_is_given() -> None:
    """Sharding correctness. The full set is collected as one process per seed and the
    analysis pools the shards, so a shard that quietly ran seed 0 instead of the seed it
    was asked for would duplicate one run ten times and look like a suspiciously tight
    result."""
    collected = SkillTraceCollector.collect(
        label="shard",
        sampler_iters=_ITERS,
        seeds=[3],
        num_cycles=1,
        max_steps=20,
        num_test_tasks=4,
    )
    assert [run["seed"] for run in collected["seeds"]] == [3]


def test_two_different_seeds_do_not_produce_identical_traces() -> None:
    """Non-vacuity for the pooling above: if the seed did not actually reach the run,
    every shard would be the same numbers and the across-seed spread would be zero."""
    runs = SkillTraceCollector.collect(
        label="pair",
        sampler_iters=_ITERS,
        seeds=[0, 1],
        num_cycles=2,
        max_steps=40,
        num_test_tasks=4,
    )["seeds"]
    assert runs[0]["periods"] != runs[1]["periods"]
