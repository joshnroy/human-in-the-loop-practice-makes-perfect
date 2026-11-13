"""Whether a resumed step-protocol run ended exactly where an uninterrupted one did.

The comparison the Della proof of Curtis used (the comparator in
`docs/experiment-logs/data/2026-11-11-tossing-resume-validation/execution-tools.tar.gz`),
kept field for field so that every other arm is held to the standard Curtis met. Two
checkpoints are read without building a simulator: live objects stay references, and a
sampler recorder is read as its saved offsets.

Three more process-local values turned up once arms other than Curtis were compared: EES
keys its seen training tasks by memory address (compared here as the tasks themselves,
in order), and expectimax and DP each keep a wall-clock duration of their own search or
solve (ignored, as the proof ignores every other timing). The two normalisations below
are the proof's own. None of the five drops a learned value:

- A belief-space method keys its cost forecasts by a `GroundSkill` repr, which embeds
  callable addresses. Each key is parsed back to (skill, bound objects) and must match
  the method's own groundings one to one; every cost is kept.
- Fast Downward's cached log text carries temporary paths, timings and peak memory. It
  is left out; the cache keys and the translated SAS payloads are compared.
"""

import ast
import collections
import enum
import functools
import hashlib
import inspect
import json
import pickle
import re
from pathlib import Path
from typing import Any, ClassVar

import numpy as np
import torch


class LiveReference:
    """A live object the checkpoint names instead of pickling (environment, predicates)."""

    def __init__(self, *, name: str) -> None:
        self.name = name

    def __getattr__(self, name: str) -> "LiveReference":  # noqa: PLR0917 -- dunder
        if name.startswith("__"):
            raise AttributeError(name)
        return LiveReference(name=f"{self.name}.{name}")


class RecorderView:
    """A sampler recorder as its checkpointed state: byte offsets, not file handles."""

    def __setstate__(self, state: Any) -> None:  # noqa: PLR0917 -- pickle API
        self.state = state


class ReadOnlyCheckpoint(pickle.Unpickler):
    def persistent_load(self, pid: Any) -> LiveReference:  # noqa: PLR0917 -- pickle API
        return LiveReference(name=pid)

    def find_class(self, module: str, name: str) -> Any:  # noqa: PLR0917 -- pickle API
        if name == "SamplerDrawRecorder":
            return RecorderView
        return super().find_class(module, name)


class ResumeEquivalence:
    # Wall-clock, paths, handles and hashes of process-dependent files: what the proof
    # normalised away. Nothing here is a belief, a weight, a cost, a count or an RNG.
    IGNORED: ClassVar[frozenset[str]] = frozenset({
        "header",
        "elapsed_seconds",
        "written_unix",
        "updated_unix",
        "output_dir",
        "output_path",
        "decision_log",
        "draw_recorder",
        "ees_decision_log",
        "policy_file_sha256",
        "policy_sha256",
        "git_dirty",
        "git_commit",
        "event_log_bytes",
        "checkpoint_written_unix",
        "seconds",
        "timing",
        "timestamp",
        # Not in the proof's list, which only met Curtis. Two more wall-clock durations:
        # one expectimax search (`time.perf_counter` differences, kept in its trace), and
        # one DP allocation (Gurobi's `Runtime`, kept with the solver's result).
        "search_elapsed_seconds",
        "runtime_seconds",
    })
    CHECKPOINT_FIELDS: ClassVar[tuple[str, ...]] = (
        "method",
        "accounting",
        "metrics",
        "cycle",
        "robot_invocations",
        "learning_updates",
        "reset_counts",
        "records",
        "evaluations",
        "clock",
        "env_state",
        "simulation_state",
        "rngs",
        "stop_streak",
        "zero_step_streak",
        "no_robot_action_streak",
    )
    # What a cycle of practice cannot leave untouched, whatever the arm does in it.
    MOVES_EVERY_CYCLE: ClassVar[tuple[str, ...]] = ("method", "cycle", "learning_updates")
    EVENTS: ClassVar[frozenset[str]] = frozenset({
        "session_start",
        "robot",
        "human",
        "stop",
        "learning_update",
        "measurement",
    })
    RECORDER_FILES: ClassVar[tuple[str, ...]] = ("sampler_draws.jsonl", "sampler_proposals.jsonl")
    QUOTED: ClassVar[str] = r"(?:'(?:\\.|[^'\\])*'|\"(?:\\.|[^\"\\])*\")"
    SKILL_NAME: ClassVar[re.Pattern[str]] = re.compile(r"^skill=Skill\(name=(" + QUOTED + r"),")
    BOUND_OBJECT: ClassVar[re.Pattern[str]] = re.compile(
        r"Object\(name=(?P<name>"
        + QUOTED
        + r"), type=Type\(name=(?P<type>"
        + QUOTED
        + r"), feature_names=(?P<features>\((?:"
        + QUOTED
        + r"(?:,\s*)?)*\))\)\)"
    )

    @staticmethod
    def load(*, path: Path) -> dict[str, Any]:
        with path.open("rb") as stream:
            pickle.load(stream)  # noqa: S301 -- the header; this test wrote the file
            return ReadOnlyCheckpoint(stream).load()

    @staticmethod
    def ground_skill_identity(*, ground: Any) -> tuple[Any, ...]:
        return (
            ground.skill.name,
            tuple(
                (obj.name, obj.type.name, tuple(obj.type.feature_names)) for obj in ground.objects
            ),
        )

    @staticmethod
    def parsed_ground_skill_identity(*, text: Any) -> tuple[Any, ...]:
        """Read only quoted literals; never evaluate arbitrary serialized expressions."""
        if not isinstance(text, str):
            raise ValueError("cost forecast key must be a GroundSkill repr string")
        skill = ResumeEquivalence.SKILL_NAME.match(text)
        if skill is None or " objects=" not in text:
            raise ValueError("malformed cost forecast GroundSkill key")
        tail = text.rsplit(" objects=", 1)[1]
        objects = []
        residue = []
        position = 0
        for match in ResumeEquivalence.BOUND_OBJECT.finditer(tail):
            residue.append(tail[position : match.start()])
            name, typename, features = (
                ast.literal_eval(match.group(field)) for field in ("name", "type", "features")
            )
            if not isinstance(features, tuple) or not all(isinstance(f, str) for f in features):
                raise ValueError("malformed bound-object feature names")
            objects.append((name, typename, features))
            position = match.end()
        residue.append(tail[position:])
        if not re.fullmatch(r"[\s(),]*", "".join(residue)):
            raise ValueError("malformed cost forecast bound-object tuple")
        return (ast.literal_eval(skill.group(1)), tuple(objects))

    @staticmethod
    def canonical_execution_costs(*, method: Any) -> dict[Any, Any]:
        """Keep every cost and require a bijection with the method's exact groundings."""
        identities = [
            ResumeEquivalence.ground_skill_identity(ground=g)
            for g in method._pomdp_model.ground_skills
        ]
        if len(identities) != len(set(identities)):
            raise ValueError("ground skills do not have unique semantic identities")
        expected = set(identities)
        result: dict[Any, Any] = {}
        for key, value in method._pomdp_state.expected_execution_costs.items():
            parsed = ResumeEquivalence.parsed_ground_skill_identity(text=key)
            if parsed not in expected:
                raise ValueError("cost forecast key matches no method ground skill")
            if parsed in result:
                raise ValueError("duplicate semantic cost forecast key")
            result[parsed] = value
        if set(result) != expected:
            raise ValueError("cost forecast keys are not a bijection with method ground skills")
        return result

    @staticmethod
    def normalize(*, value: Any, stack: frozenset[int] = frozenset()) -> Any:  # noqa: PLR0911, PLR0912
        def inner(item: Any) -> Any:  # noqa: PLR0917 -- one-argument recursion shorthand
            return ResumeEquivalence.normalize(value=item, stack=stack)

        if value is None or isinstance(value, str | bool | int):
            return value
        if isinstance(value, float):
            return value if np.isfinite(value) else repr(value)
        if isinstance(value, np.generic):
            return inner(value.item())
        if isinstance(value, np.ndarray):
            return {
                "dtype": str(value.dtype),
                "shape": list(value.shape),
                "values": inner(value.tolist()),
            }
        if isinstance(value, np.dtype):
            return {"numpy_dtype": str(value)}
        if isinstance(value, np.random.RandomState):
            return inner(value.get_state())
        if isinstance(value, collections.deque):
            return {"deque": inner(list(value)), "maxlen": value.maxlen}
        if isinstance(value, functools.partial):
            return {
                "partial": inner(value.func),
                "args": inner(value.args),
                "keywords": inner(value.keywords),
            }
        if isinstance(value, np.random.Generator):
            return inner(value.bit_generator.state)
        if isinstance(value, Path):
            return {"path_basename": value.name}
        if isinstance(value, bytes):
            return {"bytes_sha256": hashlib.sha256(value).hexdigest()}
        if isinstance(value, enum.Enum):
            return inner(value.value)
        if isinstance(value, LiveReference):
            return {"live_reference": value.name}
        if inspect.isclass(value):
            return {"class": f"{value.__module__}.{value.__qualname__}"}
        if inspect.isfunction(value) or inspect.ismethod(value) or inspect.isbuiltin(value):
            return {"callable": f"{value.__module__}.{value.__qualname__}"}
        if isinstance(value, torch.Tensor):
            return inner(value.detach().cpu().numpy())
        if id(value) in stack:
            return {"cycle_reference": f"{type(value).__module__}.{type(value).__qualname__}"}
        stack = stack | {id(value)}
        if isinstance(value, dict):
            rows = [
                (inner(k), inner(v))
                for k, v in value.items()
                if not isinstance(k, str) or k not in ResumeEquivalence.IGNORED
            ]
            return {"mapping": sorted(rows, key=lambda row: json.dumps(row[0], sort_keys=True))}
        if isinstance(value, list | tuple):
            return [inner(v) for v in value]
        if isinstance(value, set | frozenset):
            return sorted(
                (inner(v) for v in value), key=lambda row: json.dumps(row, sort_keys=True)
            )
        if hasattr(type(value), "__dataclass_fields__"):  # Third-party objects only.
            fields = {name: getattr(value, name) for name in type(value).__dataclass_fields__}
        elif hasattr(value, "__dict__"):
            fields = {**vars(value), **(getattr(value, "__pydantic_private__", None) or {})}
        elif hasattr(type(value), "__slots__"):
            slots = [
                name
                for cls in type(value).__mro__
                for name in getattr(cls, "__slots__", ())
                if name not in {"__dict__", "__weakref__"}
            ]
            fields = {name: getattr(value, name) for name in slots if hasattr(value, name)}
        else:
            raise TypeError(f"unhandled checkpoint semantic type: {type(value)!r}")
        if type(value).__name__ == "Tossing3DPomdpMethod":
            # Repr keys include callable addresses and unordered predicate reprs. The
            # method's groundings remain fully compared elsewhere; retain every cost.
            fields["_pomdp_state"] = value._pomdp_state.model_copy(
                update={
                    "expected_execution_costs": ResumeEquivalence.canonical_execution_costs(
                        method=value
                    )
                }
            )
        if isinstance(fields.get("_seen_training_tasks"), dict):
            # EES keys its seen training tasks by `id`: a memory address, different in
            # every process. Every task is kept, in the order EES first saw it.
            fields["_seen_training_tasks"] = list(fields["_seen_training_tasks"].values())
        if (type(value).__module__, type(value).__name__) == (
            "hitl_pmp.planning.types",
            "TranslationResult",
        ):
            # External FD diagnostics include temp paths, timing and peak memory. Cache
            # keys and exact SAS payload (including None for failure) remain compared.
            fields.pop("fd_output")
        return {
            "type": f"{type(value).__module__}.{type(value).__qualname__}",
            "state": inner(fields),
        }

    @staticmethod
    def fingerprint(*, value: Any) -> str:
        normal = ResumeEquivalence.normalize(value=value)
        encoded = json.dumps(normal, sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(encoded).hexdigest()

    @staticmethod
    def differences(*, left: Any, right: Any, path: str = "$") -> list[dict[str, Any]]:
        """Where two normalised values part, leaf by leaf, for a failure to name."""
        if type(left) is not type(right):
            return [{"path": path, "resumed": left, "reference": right}]
        found: list[dict[str, Any]] = []
        if isinstance(left, dict) and set(left) == set(right) == {"mapping"}:
            a = {json.dumps(k, sort_keys=True): (k, v) for k, v in left["mapping"]}
            b = {json.dumps(k, sort_keys=True): (k, v) for k, v in right["mapping"]}
            for key in sorted(set(a) | set(b)):
                label = (a.get(key) or b[key])[0]
                where = f"{path}.{label if isinstance(label, str) else key[:180]}"
                if key not in a or key not in b:
                    found.append({"path": where, "resumed": a.get(key), "reference": b.get(key)})
                else:
                    found += ResumeEquivalence.differences(
                        left=a[key][1], right=b[key][1], path=where
                    )
        elif isinstance(left, dict):
            for key in sorted(set(left) | set(right)):
                if key not in left or key not in right:
                    found.append({
                        "path": f"{path}.{key}",
                        "resumed": left.get(key),
                        "reference": right.get(key),
                    })
                else:
                    found += ResumeEquivalence.differences(
                        left=left[key], right=right[key], path=f"{path}.{key}"
                    )
        elif isinstance(left, list):
            if len(left) != len(right):
                found.append({
                    "path": f"{path}.length",
                    "resumed": len(left),
                    "reference": len(right),
                })
            for index, (x, y) in enumerate(zip(left, right, strict=False)):
                found += ResumeEquivalence.differences(left=x, right=y, path=f"{path}[{index}]")
        elif left != right:
            found.append({"path": path, "resumed": left, "reference": right})
        return found

    @staticmethod
    def rows(*, path: Path) -> list[dict[str, Any]]:
        return [json.loads(row) for row in path.read_text().splitlines()]

    @staticmethod
    def events(*, output: Path) -> list[dict[str, Any]]:
        return [
            row
            for row in ResumeEquivalence.rows(path=output / "step_events.jsonl")
            if row["event"] in ResumeEquivalence.EVENTS
        ]

    @staticmethod
    def practice_ticks(*, output: Path) -> list[tuple[int, dict[str, Any]]]:
        """Every practice control step's recorded state, in order, with its cycle: the
        run's recording, then each resumed process's own. A state is the robot's base,
        arm and gripper positions and velocities, the cube's and bin's poses and
        velocities, and the eight finger joints, at MuJoCo's float64."""
        ticks: list[tuple[int, dict[str, Any]]] = []
        for path in (
            output / "recording.jsonl",
            *sorted(output.glob("resume-*-recording.jsonl")),
        ):
            cycle, practising = None, False
            for row in ResumeEquivalence.rows(path=path):
                if row["kind"] == "segment":
                    practising = row["phase"] == "practice"
                elif row["kind"] == "segment_end":
                    practising = False
                elif row["kind"] == "cycle":
                    cycle = row["cycle"]
                elif row["kind"] == "tick" and practising and cycle is not None:
                    ticks.append((cycle, row["state"]))
        return ticks

    @staticmethod
    def compare(*, resumed: Path, reference: Path, first_boundary: Path) -> dict[str, Any]:
        """Every comparison of the Della proof. `first_boundary` holds the paused run's
        checkpoint and recorder files as they stood after its first cycle."""
        a = ResumeEquivalence.load(path=resumed / "checkpoint.pickle")
        b = ResumeEquivalence.load(path=reference / "checkpoint.pickle")
        pairs: dict[str, tuple[Any, Any]] = {
            field: (a[field], b[field]) for field in ResumeEquivalence.CHECKPOINT_FIELDS
        }
        pairs["events"] = (
            ResumeEquivalence.events(output=resumed),
            ResumeEquivalence.events(output=reference),
        )
        # Not in the proof: every control step of both cycles, so a run that drifts and
        # then happens to land on the same final checkpoint fields still fails, and the
        # first step that differs is named.
        ticks_a = ResumeEquivalence.practice_ticks(output=resumed)
        ticks_b = ResumeEquivalence.practice_ticks(output=reference)
        pairs["practice_ticks"] = (ticks_a, ticks_b)
        first_tick = next(
            (i for i, (x, y) in enumerate(zip(ticks_a, ticks_b, strict=False)) if x != y),
            None if len(ticks_a) == len(ticks_b) else min(len(ticks_a), len(ticks_b)),
        )
        rows_a = json.loads((resumed / "step_evaluations.json").read_text())
        rows_b = json.loads((reference / "step_evaluations.json").read_text())
        pairs["evaluation_json"] = (rows_a, rows_b)
        for filename in ResumeEquivalence.RECORDER_FILES:
            path_a, path_b = resumed / filename, reference / filename
            if path_a.exists() != path_b.exists():
                pairs[filename] = (path_a.exists(), path_b.exists())
            elif path_a.exists():
                pairs[filename] = (
                    ResumeEquivalence.rows(path=path_a),
                    ResumeEquivalence.rows(path=path_b),
                )
        hashes = {
            name: {
                "resumed": ResumeEquivalence.fingerprint(value=left),
                "reference": ResumeEquivalence.fingerprint(value=right),
            }
            for name, (left, right) in pairs.items()
        }
        paused = ResumeEquivalence.load(path=first_boundary / "checkpoint.pickle")
        structural = {
            "two_cycles": a["cycle"] == b["cycle"] == 2,
            "two_updates": a["learning_updates"] == b["learning_updates"] == 2,
            "eval0_once": [row["index"] for row in rows_a] == [0, 1, 2],
            "evals_complete": all(row["complete"] for row in rows_a + rows_b),
            "first_boundary": paused["cycle"] == 1,
        }
        kinds = sorted(
            {row.get("deployment_policy", {}).get("kind") for row in rows_a + rows_b}, key=str
        )
        for filename in ResumeEquivalence.RECORDER_FILES:
            first = first_boundary / filename
            if first.exists():
                structural["preserved_prefix_" + filename] = (
                    (resumed / filename).read_bytes().startswith(first.read_bytes())
                )
        # The control: one cycle short must not pass for two. A comparison that could not
        # tell the paused run from the finished one would prove nothing above.
        sensitive = {
            field: ResumeEquivalence.fingerprint(value=paused[field]) != hashes[field]["reference"]
            for field in ResumeEquivalence.MOVES_EVERY_CYCLE
        }
        mismatched = sorted(
            name for name, row in hashes.items() if row["resumed"] != row["reference"]
        )
        return {
            "differences": {
                name: ResumeEquivalence.differences(
                    left=ResumeEquivalence.normalize(value=pairs[name][0]),
                    right=ResumeEquivalence.normalize(value=pairs[name][1]),
                    path=name,
                )
                for name in mismatched
            },
            "hashes": hashes,
            "insensitive": sorted(name for name, moved in sensitive.items() if not moved),
            "structural": structural,
            "deployment_kinds": kinds,
            "practice_ticks_by_cycle": {
                str(cycle): [sum(c == cycle for c, _ in t) for t in (ticks_a, ticks_b)]
                for cycle in sorted({c for c, _ in ticks_a + ticks_b})
            },
            "first_differing_tick": first_tick,
            "mismatched": mismatched,
            "failed_structural": sorted(name for name, ok in structural.items() if not ok),
        }
