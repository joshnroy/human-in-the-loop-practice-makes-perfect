import math

from pydantic import BaseModel, Field

from hitl_pmp.core.method.types import PracticeTargetTally, SkillPracticeTally

from .types import EvaluationBreakdown, TaskOutcome


class Metrics(BaseModel):
    """Evaluation protocol -- a fully concrete, directly-usable instance now (not a
    static-method container): every method here is a genuine, reusable default, not
    a per-domain requirement, since nothing in this codebase today needs behavior
    other than what's written here -- exactly one task/goal type. This is a step
    further than Problem's own facade
    pattern (problem/problem.py): Problem still has one genuinely must-override
    method (run_task_episode); Metrics has none, so unlike Problem/Method/
    Environment/HumanOracle/Tasks/Renderer it isn't actually one of this project's
    abstract interfaces -- callers just construct Metrics() directly, with no
    per-domain/per-method subclass needed. Human-intervention cost is now genuinely
    tracked rather than hardcoded to zero (see record_human_intervention), driven by
    the harness rather than by any Method; a future multi-task environment would override
    task_training_curve_by_subtask/percentage_success_per_task_test --
    inheriting everything else unchanged either way (ordinary subclassing).

    evaluations/task_name are real instance fields now: a fresh Metrics() per
    (method, seed) run in a reproduction sweep replaces the old
    ClassVar-plus-reset() dance -- there's no shared mutable slot left to
    accidentally leak between runs or forget to reset()."""

    # Each tuple is (transitions, solved, total).
    evaluations: list[tuple[int, int, int]] = Field(default_factory=list)
    # Per-task detail behind each evaluations entry, when the caller supplies it.
    # Defaults to empty, so every stats.json written before this field existed
    # still loads, and a caller that only has aggregate counts stays valid.
    breakdowns: list[EvaluationBreakdown] = Field(default_factory=list)
    # How many times the harness put the environment back to the current practice
    # task's initial state, counted as it happened. Defaults to 0 so every
    # stats.json written before this field existed still loads.
    num_practice_resets: int = 0
    # Planning failures and planning attempts, bucketed by the window between two
    # consecutive recordings -- always the same length, and always a failures/attempts
    # pair, never a bare numerator (see record_planning_outcomes). Both default to
    # empty so every stats.json written before these fields existed still loads; a
    # run driven by method_runner.py always writes them, zeros included, so "planned
    # fine" stays distinguishable from "did not plan".
    planning_failures_per_cycle: list[int] = Field(default_factory=list)
    planning_attempts_per_cycle: list[int] = Field(default_factory=list)
    # What each lifted skill's practice actually did, bucketed by the same window as
    # the planning counters above -- see record_practice_outcomes. Defaults to empty so
    # every stats.json written before this field existed still loads; a run driven by
    # method_runner.py always writes one entry per window, `{}` included, so "practiced
    # nothing this cycle" stays distinguishable from "this Method does not measure
    # practice".
    #
    # SkillPracticeTally comes from core.method.types rather than this package's own
    # types.py, which is deliberate: it is the SAME record the Method hands over, so
    # there is one definition of the six counters and no shape to keep in sync. That is
    # the sibling-types.py import CLAUDE.md prescribes for exactly this case -- and it
    # does not compromise what TaskOutcome's docstring protects, which is that Metrics
    # depends on nothing DOMAIN-specific. core.method.types is six ints and a name.
    practice_outcomes_per_cycle: list[dict[str, SkillPracticeTally]] = Field(default_factory=list)
    # Which lifted skills practice was actually *aimed at*, bucketed by the same window
    # again. Separate from the field above because they count different events -- see
    # PracticeTargetTally and Method.practice_target_outcomes -- and specifically because
    # the field above cannot show a skill EES is declining to practice: that skill keeps
    # being executed as a prefix step, so its execution tally never drops.
    practice_target_outcomes_per_cycle: list[dict[str, PracticeTargetTally]] = Field(
        default_factory=list
    )
    # How many times a human was asked to intervene, and what those interventions cost
    # in total, counted as they happened. Both default to 0 so every stats.json written
    # before these fields existed still loads -- and so a run with no HumanOracle wired
    # at all reports exactly the zeros it reported before.
    #
    # Two scalars rather than a per-intervention list: the only HumanOracle that exists
    # (humans/oracle.py's v0) charges a flat cost, so a list would be one number repeated
    # up to `num_cycles * max_steps_per_interaction` times and would bloat every
    # stats.json for no information. They are kept apart rather than collapsed into one
    # because they come apart the moment a v1 cost model lands, and a metric that had to
    # change shape then would invalidate the comparison to these runs.
    #
    # `float`, not `core.problem.human.types.Cost`, which is an alias for exactly `float`
    # -- importing it would add a cross-subpackage dependency that buys no type
    # information. It IS that Cost; this is the same quantity a HumanOracle returns.
    num_human_interventions_recorded: int = 0
    summed_human_cost_recorded: float = 0.0
    task_name: str = "default"

    def record_evaluation(
        self,
        *,
        num_online_transitions: int,
        num_solved: int,
        num_total: int,
        outcomes: tuple[TaskOutcome, ...] | None = None,
    ) -> None:
        """Records one evaluation checkpoint (e.g. after an online-learning cycle) --
        the building block task_training_curve() reports back out.

        outcomes is optional per-task detail; when given it must agree with
        num_solved/num_total, since the aggregate stays the primary record and a
        silent disagreement between the two would be undetectable downstream."""
        breakdown = (
            None
            if outcomes is None
            else EvaluationBreakdown(
                num_online_transitions=num_online_transitions, outcomes=outcomes
            )
        )
        # Validated before either list is appended to, so a rejected call leaves
        # this Metrics untouched rather than half-updated with the aggregate.
        if breakdown is not None and (
            len(breakdown.outcomes) != num_total or breakdown.num_solved() != num_solved
        ):
            raise ValueError(
                f"per-task outcomes disagree with the aggregate: got "
                f"{breakdown.num_solved()}/{len(breakdown.outcomes)} from outcomes, "
                f"{num_solved}/{num_total} from the counts"
            )
        self.evaluations.append((num_online_transitions, num_solved, num_total))
        if breakdown is not None:
            self.breakdowns.append(breakdown)

    def record_practice_reset(self) -> None:
        """Counts one free reset back to the current practice task's initial state.

        Recorded rather than rederived from the configured interval because a
        configured knob is a claim and this is the measurement: an experiment
        that varies how often the robot is rescued has to be able to show the
        resets really happened, at the rate intended, from the run's own output
        instead of from an argument about the loop's arithmetic."""
        self.num_practice_resets += 1

    def record_human_intervention(self, *, cost: float) -> None:
        """Counts one human intervention and adds what it cost.

        Recorded rather than rederived from the configured trigger for the same reason
        `record_practice_reset` is: a flag is a claim and this is the measurement. An arm
        configured to call a human "when stuck" and an arm that never actually got stuck
        produce identical command lines, and only this number tells them apart -- which
        matters most for a null result, where "the human did not help" and "the human was
        never called" are completely different findings.

        `cost` is `core.problem.human.types.Cost` -- what
        `Problem.calculate_cost_for_human_command` returned for the command that was then
        executed. The caller passes the *queried* cost rather than recomputing one, so
        the number banked is the number the oracle actually quoted.

        Rejects a negative or non-finite cost, and rejects it before either field moves
        so a refused call leaves this Metrics untouched rather than half-updated. Non-
        finite is the sharper of the two: `Cost` is documented as `inf` when the command
        is infeasible, so an infinite cost here means a caller executed a command its own
        oracle had already declared impossible. Summing that would make
        `summed_human_cost` inf for the rest of the run and destroy every later
        comparison, which is a failure worth stopping for rather than averaging away."""
        if not math.isfinite(cost):
            raise ValueError(
                f"a human intervention's cost must be finite, got {cost}. Cost is inf "
                "when the command is infeasible, so this means a command the oracle "
                "refused to price was executed anyway."
            )
        if cost < 0:
            raise ValueError(f"a human intervention's cost cannot be negative, got {cost}")
        self.num_human_interventions_recorded += 1
        self.summed_human_cost_recorded += cost

    def record_planning_outcomes(self, *, num_failures: int, num_attempts: int) -> None:
        """Records one window's planning failures **and** the attempts they are out of.

        Recorded because a planner that never succeeds and a planner that succeeds but
        plans badly produce the *same* stats.json otherwise. A malformed-PDDL defect
        once made `EesMethod` catch `PlanningFailure` on every single step and degrade
        to a no-op for a whole run; the run exited 0 with a full stats.json reporting
        0/5, and nothing anywhere recorded that planning had failed. "The method scored
        zero" and "the method never planned" are different diagnoses, and only the
        second one is immediate.

        The denominator is not optional, and this is why it is one call rather than
        two: a bare failure count is uninterpretable here. EES asks the planner
        speculatively -- once per seen task in the planning-progress scoring pass, once
        per candidate while situating -- and a failure in those loops is *routine*, it
        just drops that task or candidate. So a perfectly healthy run reports a nonzero,
        workload-dependent number, and only `failures ≈ attempts` means anything is
        wrong. `17/20` says that; `17` does not. (This is the project's standing
        counts-not-percentages rule arriving at the same place: never record a
        numerator whose denominator the reader cannot recover.)

        **What one window is**, exactly, since it is not the obvious thing:
        method_runner.py records at `PracticeLoop`'s `on_cycle_end`, which fires after
        cycle *i*'s practice period and *before* cycle *i*'s evaluation sweep, plus once
        more after the loop. So entry `i < N` covers *evaluation sweep i then practice
        period i*, and the final entry `N` covers *evaluation sweep N alone* -- a
        smaller kind of window than the others, not a short cycle. The list is therefore
        the same length as `evaluations` but offset from it: bucket `i` includes the
        practice that runs between `evaluations[i]`'s and `evaluations[i+1]`'s
        transition counts, so plotting the two against one x-axis without accounting for
        that shifts a whole practice period's failures one checkpoint left.

        Rejects rather than clamps a negative count or a failures > attempts pair:
        callers record deltas between two cumulative readings, so either means a
        counter went backwards or the two got out of step -- a bug worth surfacing
        rather than averaging away."""
        if num_failures < 0 or num_attempts < 0:
            raise ValueError(
                f"planning counts cannot be negative: got {num_failures}/{num_attempts}"
            )
        if num_failures > num_attempts:
            raise ValueError(
                f"planning failures cannot exceed attempts: got {num_failures}/{num_attempts}"
            )
        self.planning_failures_per_cycle.append(num_failures)
        self.planning_attempts_per_cycle.append(num_attempts)

    def total_planning_outcomes(self) -> tuple[int, int]:
        """(failures, attempts) over the whole run -- an x/y pair, never a bare x."""
        return (sum(self.planning_failures_per_cycle), sum(self.planning_attempts_per_cycle))

    def record_practice_outcomes(self, *, outcomes: dict[str, SkillPracticeTally]) -> None:
        """Records one window's per-lifted-skill practice tallies.

        Recorded because a run that solved 21/100 because its samplers never got enough
        labels and a run that solved 21/100 because its samplers cannot fit the labels
        they have produce the *same* stats.json otherwise. Both of this project's two
        most recent experiments (PR #103, PR #108) ran into exactly that, and only one
        of them could answer it -- through a bespoke per-domain collector script. This
        is the same measurement made available to every Method on every domain.

        **The window is the planning counters' window**, which is not the obvious one:
        method_runner.py records at `PracticeLoop`'s `on_cycle_end`, which fires after
        cycle *i*'s practice period and *before* cycle *i*'s evaluation sweep, plus once
        more after the loop. So entry `i < N` covers *evaluation sweep i then practice
        period i*, and the final entry `N` covers the final sweep alone -- which
        contains no practice at all, so it is `{}` for every skill. The list is
        therefore the same length as `evaluations` but offset from it: bucket `i`
        holds the practice that runs between `evaluations[i]`'s and `evaluations[i+1]`'s
        transition counts. Plotting the two against one x-axis without accounting for
        that shifts a whole practice period one checkpoint left.

        **Stored with the skill names sorted.** stats.json's byte-stability is what
        verifies a change did not alter results, and a dict serializes in insertion
        order, so an unsorted window would make the file depend on which skill happened
        to be practiced first. Sorted here rather than at the caller so no caller can be
        the one that forgets.

        An empty window is appended rather than skipped: the buckets are only readable
        against `evaluations` if they stay index-aligned, and a cycle in which nothing
        was practiced is a finding, not an absence."""
        self.practice_outcomes_per_cycle.append({name: outcomes[name] for name in sorted(outcomes)})

    def total_practice_outcomes(self) -> dict[str, SkillPracticeTally]:
        """Each lifted skill's tally summed over the whole run.

        Summed by `SkillPracticeTally.plus`, so the validator that keeps a tally
        internally consistent applies to the total too -- there is no path by which the
        pools and the attempts disagree in a report but agree in the record."""
        totals: dict[str, SkillPracticeTally] = {}
        for window in self.practice_outcomes_per_cycle:
            for name, tally in window.items():
                totals[name] = totals.get(name, SkillPracticeTally()).plus(other=tally)
        return {name: totals[name] for name in sorted(totals)}

    def record_practice_target_outcomes(self, *, outcomes: dict[str, PracticeTargetTally]) -> None:
        """Records one window's per-lifted-skill practice-*target* tallies.

        Same window, same sorting, same append-an-empty-bucket rule as
        `record_practice_outcomes` -- see there for all three, and for why the offset
        between these buckets and `evaluations` is a trap worth reading about once."""
        self.practice_target_outcomes_per_cycle.append({
            name: outcomes[name] for name in sorted(outcomes)
        })

    def total_practice_target_outcomes(self) -> dict[str, PracticeTargetTally]:
        """Each lifted skill's selection tally summed over the whole run, by
        `PracticeTargetTally.plus`, so the total is validated like every window is."""
        totals: dict[str, PracticeTargetTally] = {}
        for window in self.practice_target_outcomes_per_cycle:
            for name, tally in window.items():
                totals[name] = totals.get(name, PracticeTargetTally()).plus(other=tally)
        return {name: totals[name] for name in sorted(totals)}

    def failures_by_goal(self) -> dict[str, tuple[int, int]]:
        """{goal description: (num_failed, num_total)} for the final evaluation
        sweep -- the "*which* tasks is it still failing?" view the aggregate
        curve cannot give. Empty when no per-task outcomes were recorded.

        Grouped by goal rather than listed per task because the useful unit is
        the task *family*: goal descriptions repeat across a test set, and a
        family that fails structurally (rather than by unlucky sampling) shows
        up as a whole group failing."""
        if not self.breakdowns:
            return {}
        grouped: dict[str, tuple[int, int]] = {}
        for outcome in self.breakdowns[-1].outcomes:
            failed, total = grouped.get(outcome.goal, (0, 0))
            grouped[outcome.goal] = (failed + int(not outcome.solved), total + 1)
        return grouped

    def task_training_curve(self) -> list[tuple[int, float]]:
        """(num_online_transitions, percentage_solved) pairs, in recorded order --
        e.g. Figure 4 of the "Practice Makes Perfect" paper plots exactly this,
        per approach per seed."""
        return [
            (transitions, (solved / total) if total else 0.0)
            for transitions, solved, total in self.evaluations
        ]

    def task_training_curve_by_subtask(self) -> dict[str, list[tuple[int, float]]]:
        return {self.task_name: self.task_training_curve()}

    def percentage_success_overall_test(self) -> float:
        if not self.evaluations:
            return 0.0
        _, solved, total = self.evaluations[-1]
        return solved / total if total else 0.0

    def percentage_success_per_task_test(self) -> dict[str, float]:
        return {self.task_name: self.percentage_success_overall_test()}

    def percentage_success_overall_train(self) -> float:
        """Not tracked: this reproduction only evaluates held-out test tasks
        after each online-learning cycle (matching predicators' own
        _run_testing, which only scores env.get_test_tasks())."""
        return 0.0

    def percentage_success_per_task_train(self) -> dict[str, float]:
        return {}

    def num_complete_environment_resets(self) -> int:
        return 0

    def num_human_interventions(self) -> tuple[float, int]:
        """Returns (summed cost, count); should trend down as the agent learns to
        reset itself.

        No longer trivially zero. It reported a hardcoded (0.0, 0) for as long as no
        concrete `HumanOracle` existed -- intervention was not *representable*, not
        merely unobserved -- and now reads the two fields `record_human_intervention`
        maintains. A run with no human wired still reports (0.0, 0), so nothing that
        predates this changes.

        A pair rather than either number alone, and the pair is the whole point at v0:
        `humans/oracle.py` charges a flat 1.0, so the two are proportional today and the
        cost carries no information the count does not. They separate as soon as a v1
        cost model prices a rescue by how far the robot has drifted -- and reporting only
        the count now would mean a v1 comparison had no v0 cost baseline to sit against."""
        return (self.summed_human_cost_recorded, self.num_human_interventions_recorded)

    def summed_human_cost(self) -> float:
        return self.summed_human_cost_recorded
