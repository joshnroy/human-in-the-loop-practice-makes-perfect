"""What the coding agent is told. A static-method container, never instantiated."""

from typing import ClassVar

from .types import CodedSamplerMode
from .workspace import CodingWorkspace


class CodedSamplerPrompts:
    PYTHON: ClassVar[str] = "/opt/robocode-strict/bin/python"
    CALL_SECONDS: ClassVar[float] = 60.0

    @staticmethod
    def system() -> str:
        return (
            "You are a coding agent. You write and maintain the code a robot uses to "
            "choose the continuous parameters of its skills while it learns from "
            "practice. You work in a sandbox with no network access. CLAUDE.md in the "
            "working directory describes the task, the files and the interface."
        )

    @staticmethod
    def interface(*, mode: CodedSamplerMode) -> str:
        if mode == CodedSamplerMode.SCORER:
            return """## Interface: scorer mode

Each module defines

```text
def score(sampler_inputs: list[list[float]]) -> list[float]
```

`sampler_inputs` holds one row per candidate parameter vector, laid out as
`docs/sampler_inputs.md` describes for that skill. Return one finite float per row;
higher means more likely to succeed. The robot draws about 100 candidates from its base
proposal, scores them with your function, and executes the highest-scoring one, breaking
ties uniformly at random. If more than half the candidates tie at the maximum, the robot
treats your scores as uninformative and picks a candidate uniformly at random. During
practice, with probability 0.5 the robot ignores your scores and picks uniformly at
random (exploration); during evaluation it always uses them.
"""
        return """## Interface: sampler mode

Each module defines

```text
def sample(state: dict, grounding: list[str], rng: numpy.random.Generator) -> list[float]
```

`state` maps every object name to `{feature name: value}` (see
`docs/sampler_inputs.md` for each type's features; a non-finite value is `None`).
`grounding` lists the names of the objects this execution of the skill binds, in the
skill's parameter order. Return the parameter vector, in the order and within the
closed bounds `docs/sampler_inputs.md` lists. A vector of the wrong length, with a
non-finite entry or outside the bounds is rejected, never clipped, and the robot draws
one from its base proposal instead. Parameters inside the bounds can still be
infeasible for the controller (for the toss: no legal place to stand at that
standoff); such an attempt is executed as a no-op and counts as a failure. During
practice, with probability 0.5 the robot skips your code and draws from its base
proposal (exploration); during evaluation it always uses your code.
"""

    @staticmethod
    def instructions(
        *, mode: CodedSamplerMode, skills: list[str], workspace: CodingWorkspace
    ) -> str:
        modules = "\n".join(
            f"- `{workspace.module_path(skill_name=skill)}` for `{skill}`" for skill in skills
        )
        return f"""# Skill-parameter code for a practising robot

A robot practises skills, then is evaluated, in cycles. At the end of every practice
cycle you are invoked again in this same session to update your code. When you finish,
the `samplers/` directory is frozen: that version is what the robot runs for every
decision until your next refit, in practice and in evaluation alike.

## Files
- `docs/task.md`: the task, the skills and what success means.
- `docs/skills.md`: every skill's operator (preconditions and effects).
- `docs/sampler_inputs.md` (and `.json`): each learned skill's input row, field by field,
  its parameters with units and bounds, and every object type's state features.
- `data/trajectories.jsonl`: the complete practice history so far, one event per line,
  oldest first. Rewritten at every refit; earlier lines never change.
- `data/new_since_refit_NNN.jsonl`: the events that are new at refit NNN.
- `data/code_errors.jsonl`: every failure of your code since the last refit (crashes,
  timeouts, invalid return values, disallowed imports).
- Evaluation outcomes are never recorded here.

Event fields: `index`, `kind`, `cycle`, `data`. Kinds include `practice_period_start`,
`skill_dispatched` (skill, grounding, params, whether the draw was an exploration draw,
which code version chose it, the state before), `skill_outcome` (success, the state
after), `human_reset_requested` / `human_reset_completed`, `environment_reset`,
`parameter_pool_starved` and `cycle_end`. An exploration draw (`epsilon_random`) was
uniform and ignored your code.

## What to write
{modules}

{CodedSamplerPrompts.interface(mode=mode)}
## Runtime rules
- Each call runs in a fresh process with no network and a time limit of
  {CodedSamplerPrompts.CALL_SECONDS:g} s including start-up.
- Only the `samplers/` directory exists at call time. Anything your code needs then,
  such as fitted parameters, must be a file under `samplers/`, loaded relative to
  `Path(__file__).parent`. `data/` and `docs/` are not available at call time.
- Imports are limited to numpy, scipy, sklearn and the Python standard library, plus
  your own modules under `samplers/`. Anything else is rejected and the call fails.
- If a call fails, times out or returns garbage, the robot falls back to a uniform draw
  from its base proposal, and the error appears in `data/code_errors.jsonl`.
- Test your modules with `{CodedSamplerPrompts.PYTHON}` before you finish. You may
  write analysis scripts anywhere outside `samplers/`, `docs/` and `data/`.
- Do not edit `docs/` or `data/`.
"""

    @staticmethod
    def refit(*, refit: int, cycle: int, start: int, end: int, errors: int, new_file: str) -> str:
        opening = "Read CLAUDE.md and docs/ first. " if refit == 0 else ""
        return (
            f"{opening}Refit {refit}, at the end of practice cycle {cycle}. "
            f"{end - start} new events: lines {start} to {end - 1} of "
            f"data/trajectories.jsonl, also in {new_file}. {errors} errors from your "
            "code since the last refit are in data/code_errors.jsonl. Update the modules "
            "under samplers/ and test them before you finish."
        )
