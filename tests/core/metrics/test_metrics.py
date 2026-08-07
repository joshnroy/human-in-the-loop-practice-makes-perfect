import math

import pytest

from hitl_pmp.core.method.types import PracticeTargetTally, SkillPracticeTally
from hitl_pmp.core.metrics.metrics import Metrics
from hitl_pmp.core.metrics.types import TaskOutcome


def test_record_evaluation_appends_a_checkpoint() -> None:
    metrics = Metrics()
    metrics.record_evaluation(num_online_transitions=0, num_solved=0, num_total=10)
    metrics.record_evaluation(num_online_transitions=150, num_solved=7, num_total=10)
    assert metrics.evaluations == [(0, 0, 10), (150, 7, 10)]


def test_task_training_curve_converts_solved_counts_to_percentages() -> None:
    metrics = Metrics()
    metrics.record_evaluation(num_online_transitions=0, num_solved=0, num_total=10)
    metrics.record_evaluation(num_online_transitions=150, num_solved=7, num_total=10)
    assert metrics.task_training_curve() == [(0, 0.0), (150, 0.7)]


def test_task_training_curve_handles_a_zero_total_without_dividing_by_zero() -> None:
    metrics = Metrics()
    metrics.record_evaluation(num_online_transitions=0, num_solved=0, num_total=0)
    assert metrics.task_training_curve() == [(0, 0.0)]


def test_task_training_curve_by_subtask_wraps_the_single_curve_under_task_name() -> None:
    metrics = Metrics(task_name="light_on")
    metrics.record_evaluation(num_online_transitions=0, num_solved=5, num_total=10)
    assert metrics.task_training_curve_by_subtask() == {"light_on": [(0, 0.5)]}


def test_percentage_success_overall_test_uses_the_most_recent_evaluation() -> None:
    metrics = Metrics()
    metrics.record_evaluation(num_online_transitions=0, num_solved=0, num_total=10)
    metrics.record_evaluation(num_online_transitions=150, num_solved=8, num_total=10)
    assert metrics.percentage_success_overall_test() == 0.8


def test_percentage_success_overall_test_is_zero_with_no_evaluations_yet() -> None:
    assert Metrics().percentage_success_overall_test() == 0.0


def test_percentage_success_per_task_test_wraps_the_overall_percentage() -> None:
    metrics = Metrics(task_name="light_on")
    metrics.record_evaluation(num_online_transitions=0, num_solved=3, num_total=10)
    assert metrics.percentage_success_per_task_test() == {"light_on": 0.3}


def test_train_metrics_are_not_tracked() -> None:
    metrics = Metrics()
    assert metrics.percentage_success_overall_train() == 0.0
    assert metrics.percentage_success_per_task_train() == {}


def test_complete_environment_resets_are_always_zero() -> None:
    """Still hardcoded, unlike the human counters beside it: nothing in this codebase
    performs a complete environment reset mid-run, so there is nothing to count."""
    assert Metrics().num_complete_environment_resets() == 0


def test_two_instances_do_not_share_evaluations() -> None:
    """The whole point of this refactor: no shared ClassVar/reset() dance anymore."""
    first = Metrics()
    second = Metrics()
    first.record_evaluation(num_online_transitions=0, num_solved=1, num_total=1)
    assert first.evaluations == [(0, 1, 1)]
    assert second.evaluations == []


def test_record_evaluation_without_outcomes_records_no_breakdown() -> None:
    """The per-task detail is opt-in: a caller with only aggregate counts stays
    valid, and every stats.json written before breakdowns existed still loads."""
    metrics = Metrics()
    metrics.record_evaluation(num_online_transitions=0, num_solved=3, num_total=10)
    assert metrics.breakdowns == []
    assert metrics.failures_by_goal() == {}


def test_failures_by_goal_groups_the_final_sweep_by_goal() -> None:
    metrics = Metrics()
    metrics.record_evaluation(
        num_online_transitions=100,
        num_solved=1,
        num_total=3,
        outcomes=(
            TaskOutcome(task_index=0, goal="ItemInBin(recycling, recycling_bin)", solved=False),
            TaskOutcome(task_index=1, goal="ItemInBin(recycling, recycling_bin)", solved=False),
            TaskOutcome(task_index=2, goal="ItemInBin(trash, trash_bin)", solved=True),
        ),
    )
    assert metrics.failures_by_goal() == {
        "ItemInBin(recycling, recycling_bin)": (2, 2),
        "ItemInBin(trash, trash_bin)": (0, 1),
    }


def test_failures_by_goal_reports_only_the_final_sweep() -> None:
    """The question it answers is "what is the *trained* policy still failing",
    so an earlier sweep's failures must not leak into the count."""
    metrics = Metrics()
    metrics.record_evaluation(
        num_online_transitions=0,
        num_solved=0,
        num_total=1,
        outcomes=(TaskOutcome(task_index=0, goal="g", solved=False),),
    )
    metrics.record_evaluation(
        num_online_transitions=100,
        num_solved=1,
        num_total=1,
        outcomes=(TaskOutcome(task_index=0, goal="g", solved=True),),
    )
    assert metrics.failures_by_goal() == {"g": (0, 1)}


def test_record_evaluation_rejects_outcomes_that_disagree_with_the_aggregate() -> None:
    """The aggregate stays the primary record, so a silent disagreement between
    the two would be undetectable downstream."""
    metrics = Metrics()
    with pytest.raises(ValueError, match="disagree with the aggregate"):
        metrics.record_evaluation(
            num_online_transitions=0,
            num_solved=2,
            num_total=1,
            outcomes=(TaskOutcome(task_index=0, goal="g", solved=False),),
        )
    assert metrics.evaluations == []
    assert metrics.breakdowns == []


def test_num_practice_resets_starts_at_zero_and_counts_up() -> None:
    metrics = Metrics()
    assert metrics.num_practice_resets == 0
    metrics.record_practice_reset()
    metrics.record_practice_reset()
    assert metrics.num_practice_resets == 2


def test_num_practice_resets_survives_a_round_trip_through_stats_json() -> None:
    """It has to reach the analysis side: an experiment that varies how often the
    robot is rescued verifies the manipulation by reading this back out of a run's
    own stats.json, not by trusting the flag it passed in."""
    metrics = Metrics()
    metrics.record_practice_reset()
    restored = Metrics.model_validate_json(metrics.model_dump_json())
    assert restored.num_practice_resets == 1


def test_a_stats_json_written_before_practice_resets_existed_still_loads() -> None:
    restored = Metrics.model_validate_json('{"evaluations": [[0, 1, 2]], "task_name": "default"}')
    assert restored.num_practice_resets == 0


def test_planning_outcomes_start_empty_and_record_in_order() -> None:
    metrics = Metrics()
    assert metrics.planning_failures_per_cycle == []
    assert metrics.planning_attempts_per_cycle == []
    metrics.record_planning_outcomes(num_failures=3, num_attempts=20)
    metrics.record_planning_outcomes(num_failures=0, num_attempts=11)
    assert metrics.planning_failures_per_cycle == [3, 0]
    assert metrics.planning_attempts_per_cycle == [20, 11]
    assert metrics.total_planning_outcomes() == (3, 31)


def test_recording_a_negative_planning_count_is_rejected() -> None:
    """The wiring records a *delta* between two cumulative readings, so a negative
    one means the counter went backwards -- a bug worth failing on rather than
    writing a nonsense number into stats.json."""
    with pytest.raises(ValueError, match="cannot be negative"):
        Metrics().record_planning_outcomes(num_failures=-1, num_attempts=0)


def test_recording_more_planning_failures_than_attempts_is_rejected() -> None:
    """The two counters are differenced independently, so they can only disagree this
    way if they have got out of step -- which would make every x/y in the file a lie."""
    with pytest.raises(ValueError, match="cannot exceed attempts"):
        Metrics().record_planning_outcomes(num_failures=4, num_attempts=3)


def test_a_rejected_planning_record_leaves_metrics_untouched() -> None:
    """Both lists are appended to only after validation, so a rejected call cannot
    leave them at different lengths -- which would silently misalign every later
    pair rather than failing where the bug is."""
    metrics = Metrics()
    with pytest.raises(ValueError, match="cannot exceed attempts"):
        metrics.record_planning_outcomes(num_failures=4, num_attempts=3)
    assert metrics.planning_failures_per_cycle == []
    assert metrics.planning_attempts_per_cycle == []


def test_planning_outcomes_survive_a_round_trip_through_stats_json() -> None:
    """The whole point: "the method scored zero" and "the method never planned" are
    different diagnoses, and only stats.json is read back after a sweep."""
    metrics = Metrics()
    metrics.record_planning_outcomes(num_failures=5, num_attempts=5)
    restored = Metrics.model_validate_json(metrics.model_dump_json())
    assert restored.planning_failures_per_cycle == [5]
    assert restored.planning_attempts_per_cycle == [5]


def test_a_stats_json_written_before_planning_outcomes_existed_still_loads() -> None:
    restored = Metrics.model_validate_json('{"evaluations": [[0, 1, 2]], "task_name": "default"}')
    assert restored.planning_failures_per_cycle == []
    assert restored.planning_attempts_per_cycle == []


def test_practice_outcomes_start_empty_and_record_in_order() -> None:
    metrics = Metrics()
    assert metrics.practice_outcomes_per_cycle == []
    metrics.record_practice_outcomes(
        outcomes={"ThrowTrash": SkillPracticeTally(num_attempts=8, num_successes=3)}
    )
    metrics.record_practice_outcomes(
        outcomes={
            "ThrowTrash": SkillPracticeTally(num_attempts=4, num_successes=4),
            "ThrowRecycling": SkillPracticeTally(num_attempts=2, num_successes=0),
        }
    )
    assert len(metrics.practice_outcomes_per_cycle) == 2
    assert metrics.practice_outcomes_per_cycle[0]["ThrowTrash"].num_attempts == 8
    totals = metrics.total_practice_outcomes()
    assert (totals["ThrowTrash"].num_successes, totals["ThrowTrash"].num_attempts) == (7, 12)
    assert (totals["ThrowRecycling"].num_successes, totals["ThrowRecycling"].num_attempts) == (0, 2)


def test_a_recorded_window_is_stored_with_its_skill_names_sorted() -> None:
    """stats.json's byte-stability is what verifies a change did not alter results, so
    the serialized key order must not depend on which skill happened to be practiced
    first."""
    metrics = Metrics()
    metrics.record_practice_outcomes(
        outcomes={
            "Zeta": SkillPracticeTally(num_attempts=1),
            "Alpha": SkillPracticeTally(num_attempts=1),
        }
    )
    assert list(metrics.practice_outcomes_per_cycle[0]) == ["Alpha", "Zeta"]


def test_recording_an_empty_window_keeps_the_buckets_aligned() -> None:
    """A cycle in which nothing was practiced still gets an entry, so the list stays
    the same length as `evaluations` instead of silently skipping a window."""
    metrics = Metrics()
    metrics.record_practice_outcomes(outcomes={})
    assert metrics.practice_outcomes_per_cycle == [{}]
    assert metrics.total_practice_outcomes() == {}


def test_practice_outcomes_survive_a_round_trip_through_stats_json() -> None:
    metrics = Metrics()
    metrics.record_practice_outcomes(
        outcomes={
            "ThrowRecycling": SkillPracticeTally(
                num_attempts=6,
                num_successes=2,
                num_random_attempts=3,
                num_random_successes=1,
                num_informed_attempts=2,
                num_informed_successes=1,
            )
        }
    )
    restored = Metrics.model_validate_json(metrics.model_dump_json())
    assert restored.practice_outcomes_per_cycle == metrics.practice_outcomes_per_cycle


def test_a_stats_json_written_before_practice_outcomes_existed_still_loads() -> None:
    restored = Metrics.model_validate_json('{"evaluations": [[0, 1, 2]], "task_name": "default"}')
    assert restored.practice_outcomes_per_cycle == []
    assert restored.total_practice_outcomes() == {}
    assert restored.practice_target_outcomes_per_cycle == []
    assert restored.total_practice_target_outcomes() == {}


def test_practice_target_outcomes_start_empty_and_record_in_order() -> None:
    metrics = Metrics()
    assert metrics.practice_target_outcomes_per_cycle == []
    metrics.record_practice_target_outcomes(
        outcomes={"ThrowTrash": PracticeTargetTally(num_scored=3, num_selected=2)}
    )
    metrics.record_practice_target_outcomes(
        outcomes={"ThrowTrash": PracticeTargetTally(num_declined_perfect=4)}
    )
    assert len(metrics.practice_target_outcomes_per_cycle) == 2
    assert metrics.practice_target_outcomes_per_cycle[0]["ThrowTrash"].num_selected == 2
    totals = metrics.total_practice_target_outcomes()
    assert totals["ThrowTrash"].num_scored == 3
    assert totals["ThrowTrash"].num_declined_perfect == 4


def test_practice_target_outcomes_are_stored_in_sorted_skill_order() -> None:
    metrics = Metrics()
    metrics.record_practice_target_outcomes(
        outcomes={
            "Zeta": PracticeTargetTally(num_scored=1),
            "Alpha": PracticeTargetTally(num_scored=1),
        }
    )
    assert list(metrics.practice_target_outcomes_per_cycle[0]) == ["Alpha", "Zeta"]


def test_human_intervention_counts_start_at_zero() -> None:
    """A run with no HumanOracle wired reports exactly what it reported before this
    field existed, so nothing that predates the human ladder changes."""
    metrics = Metrics()
    assert metrics.num_human_interventions() == (0.0, 0)
    assert metrics.summed_human_cost() == 0.0


def test_record_human_intervention_counts_and_sums() -> None:
    metrics = Metrics()
    metrics.record_human_intervention(cost=1.0)
    metrics.record_human_intervention(cost=2.5)
    assert metrics.num_human_interventions() == (3.5, 2)
    assert metrics.summed_human_cost() == 3.5


def test_a_zero_cost_intervention_still_counts() -> None:
    """The count and the cost are separate measurements: a free human is still a human,
    and an arm that was rescued 40 times for nothing is not an arm that was never
    rescued."""
    metrics = Metrics()
    metrics.record_human_intervention(cost=0.0)
    assert metrics.num_human_interventions() == (0.0, 1)


def test_record_human_intervention_rejects_a_negative_cost() -> None:
    metrics = Metrics()
    with pytest.raises(ValueError, match="negative"):
        metrics.record_human_intervention(cost=-1.0)
    assert metrics.num_human_interventions() == (0.0, 0)


def test_record_human_intervention_rejects_an_infinite_cost() -> None:
    """Cost is inf exactly when the command is infeasible, so recording one means an
    impossible command was executed -- and summing it would make every later comparison
    inf."""
    metrics = Metrics()
    with pytest.raises(ValueError, match="finite"):
        metrics.record_human_intervention(cost=math.inf)
    assert metrics.num_human_interventions() == (0.0, 0)


def test_human_interventions_survive_a_round_trip_through_stats_json() -> None:
    metrics = Metrics()
    metrics.record_human_intervention(cost=1.0)
    restored = Metrics.model_validate_json(metrics.model_dump_json())
    assert restored.num_human_interventions() == (1.0, 1)


def test_a_stats_json_predating_human_fields_still_loads() -> None:
    restored = Metrics.model_validate_json('{"evaluations": [[0, 1, 2]]}')
    assert restored.num_human_interventions() == (0.0, 0)
