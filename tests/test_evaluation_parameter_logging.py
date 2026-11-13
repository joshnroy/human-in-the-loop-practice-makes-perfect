"""An evaluation's action records carry the exact parameters and how they were chosen.

The label rounds a parameter to two decimals, and a recording's action record had
`"params": []`, so which parameter an evaluation deployed, and whether the sampler
ranked the candidates or drew among a tie, could not be read back from a run."""

import json
import random
import re
from pathlib import Path

import numpy as np
import pytest
import torch

from hitl_pmp.methods.practice_makes_perfect.ees_method import EesMethod
from hitl_pmp.recording.run_record.types import RecordingHeader
from hitl_pmp.recording.run_record.writer import RunRecordWriter
from hitl_pmp.step_protocol import RECORDING_FILENAME, StepEvaluation
from tests.step_evaluation_fakes import LightSwitchEvaluationProtocol

_VOLATILE = {"timestamp", "elapsed_seconds"}


def _evaluate(*, tmp_path: Path, name: str) -> Path:
    snapshot = tmp_path / "snapshot.pickle"
    if not snapshot.exists():
        LightSwitchEvaluationProtocol.write_snapshot(path=snapshot)
    output = tmp_path / name
    # Every action also draws from the process-global generators and traces the draws,
    # so two evaluations in one process are comparable only from the same global seeds.
    np.random.seed(0)  # noqa: NPY002 -- the global generator is the one traced
    random.seed(0)
    torch.manual_seed(0)
    StepEvaluation.run(
        snapshot=str(snapshot),
        configuration={"seed": 0, "num_test_tasks": 4, "evaluation_control_steps": 12},
        output=str(output),
        environment=LightSwitchEvaluationProtocol,
    )
    return output


def _lines(*, path: Path) -> list[dict]:
    return [
        {key: value for key, value in json.loads(line).items() if key not in _VOLATILE}
        for line in path.read_text().splitlines()
    ]


def _actions(*, output: Path) -> list[dict]:
    return [r for r in _lines(path=output / RECORDING_FILENAME) if r.get("kind") == "action"]


def test_a_parameterised_evaluation_action_records_its_exact_parameters_and_selection(*, tmp_path):
    actions = _actions(output=_evaluate(tmp_path=tmp_path, name="logged"))
    toggles = [a for a in actions if "TurnOnLight" in a["label"]]
    assert toggles
    for action in toggles:
        (shown,) = re.findall(r"params=\[([^\]]*)\]", action["label"])
        assert len(action["params"]) == 1
        # The label's two decimals are the same number, rounded; the record is not.
        assert float(shown) == pytest.approx(action["params"][0], abs=0.005)
        selection = action["parameter_selection"]
        assert selection["selection"] in {"informed", "uninformative_tie", "untrained", "random"}
        assert selection["consultation"] in {"informed", "uninformative", "epsilon_random"}
        assert isinstance(selection["sampler_fitted"], bool)
        assert selection["num_candidates"] >= 1
        assert selection["score_min"] <= selection["score"] <= selection["score_max"]
        # Evaluation never explores, and an unfitted sampler is named as such.
        assert selection["selection"] != "random"
        assert (selection["selection"] == "untrained") == (not selection["sampler_fitted"])
    assert any(a["params"][0] != round(a["params"][0], 2) for a in toggles)


def test_an_evaluation_action_without_parameters_records_no_selection(*, tmp_path):
    actions = _actions(output=_evaluate(tmp_path=tmp_path, name="logged"))
    plain = [a for a in actions if "TurnOnLight" not in a["label"]]
    assert plain
    for action in plain:
        assert action["params"] == []
        assert "parameter_selection" not in action


def test_logging_the_selection_changes_no_evaluation_action(*, tmp_path, monkeypatch):
    logged = _evaluate(tmp_path=tmp_path, name="logged")
    with monkeypatch.context() as patch:
        patch.setattr(EesMethod, "_note_parameter_choice", lambda self, **kwargs: None)
        silent = _evaluate(tmp_path=tmp_path, name="silent")
    assert all("parameter_selection" not in a for a in _actions(output=silent))
    # actions.jsonl traces the method's parameters and the global numpy, random and
    # torch draws after every action, so one extra draw anywhere would show.
    assert _lines(path=logged / "actions.jsonl") == _lines(path=silent / "actions.jsonl")
    assert (
        json.loads((logged / "results.json").read_text())["tasks"]
        == json.loads((silent / "results.json").read_text())["tasks"]
    )
    stripped = [
        {k: v for k, v in a.items() if k != "parameter_selection"} | {"params": []}
        for a in _actions(output=logged)
    ]
    assert stripped == _actions(output=silent)


def test_a_record_written_without_a_selection_is_what_it_was_before(*, tmp_path):
    path = tmp_path / "recording.jsonl"
    writer = RunRecordWriter(output_path=path, header=RecordingHeader(environment="none"))
    writer.robot_action(label="Sweep(robot)", start_step=0, end_step=3, success=True)
    writer.robot_action(
        label="Sweep(robot), params=[0.05]",
        start_step=3,
        end_step=6,
        success=True,
        params=[0.0512345678],
        parameter_selection={"selection": "informed", "score": 0.9},
    )
    writer.close()
    first, second = [r for r in _lines(path=path) if r.get("kind") == "action"]
    assert "parameter_selection" not in first and first["params"] == []
    assert second["params"] == [0.0512345678]
    assert second["parameter_selection"] == {"selection": "informed", "score": 0.9}
