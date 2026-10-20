"""The skill server's contract, as this package reads it.

`SKILL_SERVER_OBSERVE` and the skill replies are shaped exactly like prpl-tidybot's
skill server API of 2026-10-20 (v2, prpl-tidybot 35d785c): its keys, nesting and value types, in the
map frame, with the barrier and the bin as `Barrier.as_dict()` and `BinFit.as_dict()`
produce them. The client test
serves a stub over the same `BaseManager` RPC the robot uses.
"""

import contextlib
import math
import threading
from multiprocessing.managers import BaseManager
from pathlib import Path
from typing import Any

import pytest

from hitl_pmp.environments.tossing3d.real.contract import (
    AUTHKEY_ENV_VAR,
    CanPickReply,
    ObserveReply,
    PickReply,
    SkillServerClient,
    TossReply,
)

SKILL_SERVER_OBSERVE: dict[str, Any] = {
    "robot": [1.326, -0.493, -3.132],
    "cube": [0.71, -0.58, 0.31],
    "bin": {"x": 0.24, "y": 1.40, "yaw": 0.02, "length": 0.39, "width": 0.29, "height": 0.21},
    "holding": False,
    "gripper": 0.0,
    "gripper_command": 0.0,
    "cube_upright": True,
    "cube_in_bin": False,
    "barrier": {
        "x": 0.0,
        "y": 0.374,
        "length": 6.0,
        "thickness": 0.05,
        "yaw": 0.026694,
        "robot_side": [0.026690, -0.999644],
    },
    "barrier_x": 0.0,
    "cube_inside_bin": False,
    "toss_direction": -1.5707963267948966,
}


def test_an_observe_reply_parses_every_field_the_backend_reads() -> None:
    reply = ObserveReply.model_validate(SKILL_SERVER_OBSERVE)
    assert reply.robot == (1.326, -0.493, -3.132)
    assert reply.cube == (0.71, -0.58, 0.31)
    assert reply.bin is not None and reply.bin.length == 0.39
    assert reply.barrier is not None and reply.barrier.thickness == 0.05
    assert reply.barrier.robot_side == (0.026690, -0.999644)
    assert reply.cube_upright and not reply.holding and not reply.cube_in_bin
    assert reply.toss_direction == pytest.approx(-math.pi / 2)


def test_an_observe_reply_with_nothing_seen_parses() -> None:
    reply = ObserveReply.model_validate({
        **SKILL_SERVER_OBSERVE,
        "cube": None,
        "bin": None,
        "cube_upright": False,
        "toss_direction": None,
    })
    assert reply.cube is None and reply.bin is None


def test_an_observe_reply_missing_a_field_is_refused() -> None:
    broken = dict(SKILL_SERVER_OBSERVE)
    del broken["cube_in_bin"]
    with pytest.raises(ValueError, match="cube_in_bin"):
        ObserveReply.model_validate(broken)


def test_the_skill_replies_parse() -> None:
    can_pick = CanPickReply.model_validate({
        "feasible": False,
        "note": "no cube found",
        "refused": True,
        "reason": "no cube found",
    })
    assert can_pick.refused and not can_pick.feasible
    pick = PickReply.model_validate({
        "held": False,
        "gripper": 0.0,
        "note": "cube past the barrier",
        "refused": True,
        "reason": "cube past the barrier",
        "cube": {"x": 0.24, "y": 1.40, "yaw_deg": 12.0, "in_bin": True, "iou": None},
        "bin": SKILL_SERVER_OBSERVE["bin"],
        "stand": None,
        "attempts": [],
    })
    assert pick.refused and not pick.held
    toss = TossReply.model_validate({
        "tossed": True,
        "landed": [0.25, 1.41],
        "in_bin": True,
        "note": None,
        "inside_bin": True,
        "pose": [0.24, 0.01, 1.5707963],
        "direction": -1.5707963,
        "bin": SKILL_SERVER_OBSERVE["bin"],
        "swing": {"release_time": 0.675},
    })
    assert toss.landed == (0.25, 1.41) and toss.in_bin and toss.inside_bin


class _StubSkills:
    """The server side of the RPC: records calls, answers like the robot."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, tuple[Any, ...]]] = []

    def observe(self) -> dict[str, Any]:
        self.calls.append(("observe", ()))
        return SKILL_SERVER_OBSERVE

    def toss(self, standoff: float, speed: float, release_ms: float, direction: float) -> Any:  # noqa: PLR0917 -- the RPC is positional
        self.calls.append(("toss", (standoff, speed, release_ms, direction)))
        if standoff < 0.9:
            raise ValueError("standoff 0.50 outside [0.9, 1.45]")
        return {"tossed": True, "landed": [0.2, 1.4], "in_bin": True, "note": None}

    def stop(self) -> None:
        self.calls.append(("stop", ()))

    def get_calls(self) -> list[tuple[str, tuple[Any, ...]]]:
        return list(self.calls)


class _StubManager(BaseManager):
    pass


def _serve(*, server: Any) -> None:
    # `serve_forever` ends with `sys.exit(0)` once its stop event is set.
    with contextlib.suppress(SystemExit):
        server.serve_forever()


@pytest.fixture
def stub_server() -> Any:
    skills = _StubSkills()
    _StubManager.register("Skills", callable=lambda: skills)
    manager = _StubManager(address=("127.0.0.1", 0), authkey=b"test-key")
    server = manager.get_server()
    thread = threading.Thread(target=lambda: _serve(server=server), daemon=True)
    thread.start()
    yield server.address, skills
    server.stop_event.set()


def test_the_client_calls_the_server_over_base_manager_rpc(*, stub_server: Any) -> None:
    (host, port), skills = stub_server
    client = SkillServerClient(host=host, port=port, authkey=b"test-key")
    assert client.observe() == SKILL_SERVER_OBSERVE
    reply = client.toss(standoff=1.2, speed=120.0, release_ms=675.0, direction=-math.pi / 2)
    assert reply["in_bin"] is True
    assert skills.calls[-1] == ("toss", (1.2, 120.0, 675.0, -math.pi / 2))


def test_a_server_refusal_reaches_the_caller_as_a_value_error(*, stub_server: Any) -> None:
    (host, port), _ = stub_server
    client = SkillServerClient(host=host, port=port, authkey=b"test-key")
    with pytest.raises(ValueError, match="outside"):
        client.toss(standoff=0.5, speed=120.0, release_ms=675.0, direction=0.0)


def test_the_authkey_comes_from_the_environment_or_a_file(
    *, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(AUTHKEY_ENV_VAR, "from-env")
    assert SkillServerClient.read_authkey(path=None) == b"from-env"
    key_file = tmp_path / "key"
    key_file.write_text("from-file\n")
    assert SkillServerClient.read_authkey(path=key_file) == b"from-file"
    monkeypatch.delenv(AUTHKEY_ENV_VAR)
    with pytest.raises(ValueError, match=AUTHKEY_ENV_VAR):
        SkillServerClient.read_authkey(path=None)
