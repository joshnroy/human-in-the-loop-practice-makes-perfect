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
from hitl_pmp.environments.tossingroomsplitidentity.environment import (
    TossingRoomSplitIdentityEnvironment,
)
from hitl_pmp.methods.practice_makes_perfect.ees_method import EesMethod
from hitl_pmp.practice_loop import PracticeLoop
from scripts.tossingroomsplit_skill_traces import (
    CAUSAL_ENV,
    IDENTITY_ENV,
    PeriodLog,
    SkillTraceCollector,
    ThrowObservation,
    ThrowTarget,
    TracingEesMethod,
)

# Small but real: enough cycles for the collector to have drained more than once and for
# both throws to have been reached, and cheap enough to keep the suite fast. It was 3
# until the throw-representation change moved the training stream (`build_task` now draws
# four uniforms per task, not two): seed 0's first three practice tasks are now
# trash/empty/empty, so `ThrowRecycling` was never reached and the non-vacuity assertions
# below went quiet. 4 is the smallest value that reaches a recycling task again.
_CYCLES = 4
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
    `prefilled` separately from what EES scored, as the standing check that it stays
    closed -- and because `landed` re-implements the dynamics' own condition, so it has to
    include the capacity refusal or it would silently report a landing for a throw that
    never released anything.
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
            trash_weight=0.9,
            recycling_weight=0.9,
            trash_bin_distance=2.0,
            recycling_bin_distance=2.0,
            trash_count=1,
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


class TestTheForceEachSamplerChose:
    """`attempts` says how often a sampler was asked; it cannot say WHAT it answered.

    The mechanism this experiment is about is a sampler that has convinced itself of a
    wrong force and gets one datapoint per practice period to unconvince it. Distinguishing
    that from noise needs the chosen force itself, next to the target it was aiming at --
    a run of consecutive periods at a stable, badly wrong force is a different fact from
    the same landing rate produced by scatter.

    Only the LEARNED-sampler draws are recorded. An epsilon-random force is a coin flip
    and says nothing about what the sampler believes, so pooling the two would wash out
    exactly the signal being looked for."""

    @staticmethod
    def test_each_greedy_throw_records_the_force_it_chose_and_the_target_it_aimed_at() -> None:
        trace = _traced()
        seen = 0
        for period in trace["periods"]:
            for name, record in period["skills"].items():
                if not name.startswith("Throw"):
                    continue
                forces = record["greedy_forces"]
                targets = record["greedy_targets"]
                assert len(forces) == len(targets), name
                assert len(forces) == record["attempts"] - record["random_attempts"], name
                assert all(0.0 <= force <= 1.0 for force in forces), name
                # The required force spans [0.1, 0.9] -- it is derived from the bin's
                # throw_distance and the item's weight, not drawn as a U(0.5, 1.0)
                # `target_force` feature.
                assert all(0.1 <= target <= 0.9 for target in targets), name
                seen += len(forces)
        # Non-vacuity: a run in which no throw was ever chosen greedily would satisfy
        # every assertion above.
        assert seen > 0

    @staticmethod
    def test_an_epsilon_random_draw_is_not_recorded_as_a_sampler_choice() -> None:
        """Non-vacuity for the split, constructed rather than sampled: the same throw
        recorded once greedily and once as a random draw must leave exactly one force
        behind."""
        log = PeriodLog()
        log.record(
            name="ThrowRecycling",
            success=False,
            was_random=False,
            throw=ThrowObservation(landed=False, prefilled=False, force=0.02, target=0.83),
        )
        log.record(
            name="ThrowRecycling",
            success=True,
            was_random=True,
            throw=ThrowObservation(landed=True, prefilled=False, force=0.83, target=0.83),
        )
        tally = log.skills["ThrowRecycling"]
        assert tally.attempts == 2
        assert tally.random_attempts == 1
        assert tally.greedy_forces == [0.02]
        assert tally.greedy_targets == [0.83]


class TestWhichGreedyDrawsTheClassifierActuallyInformed:
    """A greedy draw is not automatically a *learned* one.

    `LearnedSkillSampler.sample` falls back to a uniform draw whenever its scores fail
    to rank the candidates -- unfitted, either single-class shortcut, or a saturated
    plateau. Those draws report `was_random=False`, so before this split they were
    pooled with the trained classifier's choices and reported as evidence of what the
    sampler had learned. `informed_*` is the subset that really is."""

    @staticmethod
    def test_an_uninformed_greedy_draw_counts_as_greedy_but_not_as_informed() -> None:
        log = PeriodLog()
        log.record(
            name="ThrowRecycling",
            success=False,
            was_random=False,
            throw=ThrowObservation(
                landed=False, prefilled=False, force=0.02, target=0.83, informed=False
            ),
        )
        log.record(
            name="ThrowRecycling",
            success=True,
            was_random=False,
            throw=ThrowObservation(
                landed=True, prefilled=False, force=0.83, target=0.83, informed=True
            ),
        )
        tally = log.skills["ThrowRecycling"]
        assert tally.greedy_forces == [0.02, 0.83]
        assert tally.informed_attempts == 1
        assert tally.informed_successes == 1
        assert tally.informed_landed == 1
        assert tally.informed_forces == [0.83]
        assert tally.informed_targets == [0.83]

    @staticmethod
    def test_every_throw_records_its_own_target_landing_and_kind_of_draw() -> None:
        """The `greedy_*`/`informed_*` lists are per-period COUNTS plus the greedy pool's
        forces and targets. Neither is enough for the scaling question.

        What the classifier can represent is set by its POSITIVES -- every landed attempt,
        epsilon-random ones included, since `observe_outcome` feeds them all back. One
        success pins where the good force region sits for one target; only two successes
        at well-separated targets reveal the slope of the force/target relation. So the
        separation among the targets of the landings is the mechanism variable, and the
        greedy lists exclude the random draws by construction -- the targets of exactly
        the landings that are most numerous early on are the ones they drop.

        These three lists are positionally aligned, one entry per throw attempt in
        execution order, and every existing count re-derives from them."""
        log = PeriodLog()
        for force, landed, random, informed in (
            (0.20, False, False, True),
            (0.83, True, False, True),
            (0.55, True, True, False),
            (0.41, False, False, False),
        ):
            log.record(
                name="ThrowTrash",
                success=landed,
                was_random=random,
                throw=ThrowObservation(
                    landed=landed, prefilled=False, force=force, target=0.83, informed=informed
                ),
            )
        tally = log.skills["ThrowTrash"]
        assert tally.throw_landed_flags == [False, True, True, False]
        assert tally.throw_targets == [0.83, 0.83, 0.83, 0.83]
        assert tally.throw_kinds == ["informed", "informed", "random", "fallback"]
        # The counts the previous log reported re-derive from the per-draw lists.
        assert sum(tally.throw_landed_flags) == tally.landed
        assert tally.throw_kinds.count("random") == tally.random_attempts
        assert tally.throw_kinds.count("informed") == tally.informed_attempts

    @staticmethod
    def test_an_epsilon_random_draw_is_never_counted_as_informed() -> None:
        log = PeriodLog()
        log.record(
            name="ThrowTrash",
            success=True,
            was_random=True,
            throw=ThrowObservation(
                landed=True, prefilled=False, force=0.5, target=0.5, informed=False
            ),
        )
        assert log.skills["ThrowTrash"].informed_attempts == 0

    @staticmethod
    def test_a_real_run_records_informed_draws_as_a_subset_of_its_greedy_ones() -> None:
        trace = _traced()
        seen_greedy = 0
        seen_informed = 0
        for period in trace["periods"]:
            for name, record in period["skills"].items():
                if not name.startswith("Throw"):
                    continue
                greedy = record["attempts"] - record["random_attempts"]
                assert record["informed_attempts"] <= greedy, name
                assert len(record["informed_forces"]) == record["informed_attempts"], name
                assert len(record["informed_targets"]) == record["informed_attempts"], name
                assert record["informed_successes"] <= record["informed_attempts"], name
                assert record["informed_landed"] <= record["informed_attempts"], name
                kinds = record["throw_kinds"]
                assert len(kinds) == record["attempts"], name
                assert len(record["throw_targets"]) == record["attempts"], name
                assert len(record["throw_landed_flags"]) == record["attempts"], name
                assert sum(record["throw_landed_flags"]) == record["landed"], name
                assert kinds.count("informed") == record["informed_attempts"], name
                assert kinds.count("random") == record["random_attempts"], name
                seen_greedy += greedy
                seen_informed += record["informed_attempts"]
        # Non-vacuity, and the claim the split exists for: this fixture's 4 cycles
        # never buy either throw sampler a classifier that discriminates, so every one
        # of its greedy draws is the uniform fallback. The two pools genuinely differ,
        # which is exactly what pooling them hid.
        assert seen_greedy > 0
        assert seen_informed < seen_greedy


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


class TestTheCollectorServesBothThrowRepresentations:
    """The collector drives two arms: `tossingroomsplit` (the CAUSAL representation, the
    default and every pre-existing caller's behaviour) and `tossingroomsplitidentity`
    (the degenerate IDENTITY one, where the required force IS `item.target_force` and
    sits at index 4 of each throw's own classifier row).

    They are traced by one script rather than two so the shards are provably the same
    SHAPE -- the experiment lays the two arms side by side, and a shape difference would
    make that unreadable. `ThrowTarget` is the only place the two differ here, which is
    the same claim the domains themselves make."""

    def test_the_identity_arm_produces_a_shard_of_the_same_shape(self) -> None:
        run = SkillTraceCollector.run_seed(
            env_name=IDENTITY_ENV,
            seed=0,
            sampler_iters=_ITERS,
            num_cycles=_CYCLES,
            max_steps=_STEPS,
            num_test_tasks=_TEST_TASKS,
        )
        causal = _traced()
        assert set(run) == set(causal)
        assert len(run["sweeps"]) == len(causal["sweeps"])
        assert set(run["sweeps"][0]) == set(causal["sweeps"][0])
        # ...and at THIS scale the two shards are not merely the same shape, they are
        # identical, which is the pairing showing through rather than a bug. The arms
        # draw the same tasks with the same required forces and consume the same RNG, so
        # the only thing that can separate them is what each classifier learns -- and at
        # 4 cycles and 100 training iterations neither discriminates, so both samplers
        # are taking `sample`'s uniform fallback and both make the same throws.
        #
        # It is asserted rather than merely observed because it is the sharpest available
        # statement of what this fork does and does not change: NOTHING outside the
        # classifier's view differs. The arms separate at experiment scale (2500
        # transitions), where the identity arm's `ThrowRecycling` lands 36/56 of its
        # informed draws against the causal arm's 11/56.
        assert run["periods"] == causal["periods"]
        assert run["sweeps"] == causal["sweeps"]

    def test_a_shard_records_which_arm_produced_it(self) -> None:
        """So a pooled analysis cannot silently mix the two. They are the same world
        under two representations: comparable side by side, never summed."""
        collected = SkillTraceCollector.collect(
            label="identity",
            env_name=IDENTITY_ENV,
            sampler_iters=_ITERS,
            seeds=[0],
            num_cycles=1,
            max_steps=20,
            num_test_tasks=4,
        )
        assert collected["env"] == IDENTITY_ENV
        assert (
            SkillTraceCollector.collect(
                label="causal",
                sampler_iters=_ITERS,
                seeds=[0],
                num_cycles=1,
                max_steps=20,
                num_test_tasks=4,
            )["env"]
            == CAUSAL_ENV
        ), "the default must stay the causal arm, so pre-existing callers are unchanged"

    def test_the_identity_arms_target_is_literally_the_items_own_feature(self) -> None:
        """The whole point of that arm, at the one place this script reads the target:
        no relation is applied, the answer is read straight out of the State."""
        env = TossingRoomSplitIdentityEnvironment()
        state = env.build_initial_state(trash_target_force=0.37, recycling_target_force=0.62)
        assert ThrowTarget.of(env=env, state=state, item=env.trash, bin_obj=env.trash_bin) == 0.37
        assert (
            ThrowTarget.of(env=env, state=state, item=env.recycling, bin_obj=env.recycling_bin)
            == 0.62
        )

    def test_the_causal_arms_target_is_not_any_single_state_feature(self) -> None:
        """Non-vacuity for the test above: the same call on the causal arm combines two
        features with coefficients that are not in the State at all, so it equals
        neither of them."""
        env = TossingRoomSplitEnvironment()
        state = env.build_initial_state(
            trash_weight=1.4,
            recycling_weight=0.6,
            trash_bin_distance=2.8,
            recycling_bin_distance=1.2,
        )
        target = ThrowTarget.of(env=env, state=state, item=env.trash, bin_obj=env.trash_bin)
        assert target == env.required_force(throw_distance=2.8, item_weight=1.4)
        assert target not in (1.4, 2.8)
