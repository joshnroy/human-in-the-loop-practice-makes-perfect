"""New simulated checkpoints define every process-global RNG from the run seed."""

import random
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from hitl_pmp.step_protocol import StepProtocolEntry
from tests.test_step_protocol import FakeStepProtocol, _step_args


class SupportedSimulation(FakeStepProtocol):
    simulation_checkpoints = True


class InProcessDomain(SupportedSimulation):
    @staticmethod
    def evaluation_pool():
        return object()


def seed_entropy(*, seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)


def invoke(*, output, entropy, seed=7, checkpoint=True, environment=SupportedSimulation):
    seed_entropy(seed=entropy)
    results = []

    def dispatched(**kwargs):
        results.append((np.random.random(), random.random(), torch.rand(1).item()))

    StepProtocolEntry.run(
        args=_step_args(
            output_dir=output, budget=8, interval=100, seed=seed, checkpoint_cycles=checkpoint
        ),
        method_cli=SimpleNamespace(run=dispatched),
        env_cli=SimpleNamespace(__name__="FakeCli"),
        environment=environment,
    )
    return results[0]


def test_fresh_checkpoint_runs_seed_all_global_streams_before_construction(*, tmp_path):
    a = invoke(output=tmp_path, entropy=123, seed=7)
    b = invoke(output=tmp_path, entropy=987, seed=7)
    c = invoke(output=tmp_path, entropy=987, seed=8)
    assert a == b
    assert a != c


@pytest.mark.parametrize(
    ("checkpoint", "environment"), [(False, SupportedSimulation), (True, InProcessDomain)]
)
def test_existing_simulation_and_in_process_initialization_is_unchanged(
    *, tmp_path, checkpoint, environment
):
    seed_entropy(seed=123)
    expected = (np.random.random(), random.random(), torch.rand(1).item())
    assert (
        invoke(output=tmp_path, entropy=123, checkpoint=checkpoint, environment=environment)
        == expected
    )
