import json
import math
from typing import Any

from hitl_pmp.methods.pure_agent.types import SkillChoice


class DecisionParseError(Exception):
    """The agent's reply could not be turned into a legal decision.

    Carried as a string into the ledger rather than raised out of the run: a malformed
    reply is a real failure mode of an agent-as-policy baseline and the run's job is to
    measure how often it happens, not to abort on the first one. The message is the whole
    diagnosis, so it is written to be read by a human scanning a ledger."""


class DecisionParser:
    """Turns one agent reply into a `SkillChoice`, or says exactly why it could not. A
    static-method container, never instantiated, same as every other business-logic class
    in this project.

    **Lenient about packaging, strict about content.** The agent is told to reply with one
    line of JSON and nothing else, and mostly does; when it does not, the failure is
    almost always a fenced block or a sentence wrapped around a perfectly good object.
    Rejecting those would measure the prompt's ability to suppress prose rather than the
    agent's ability to choose an action, so the JSON object is extracted from wherever it
    sits. What is *not* forgiven is anything that would change which action is taken: an
    out-of-range index, a parameter vector of the wrong width, a non-finite parameter. Each
    of those is a decision the harness cannot execute, and quietly repairing one would
    invent a choice the agent did not make.

    **No retry.** A reply that cannot be parsed is counted and becomes a no-op step,
    rather than triggering a second query with the error appended. Re-asking would fold
    the parser's leniency into the measured result -- an arm that re-asks until it gets a
    parseable answer reports the success rate of a policy that never malforms, which is
    not the policy that ran. The malformed count is a result, so it is left visible."""

    @staticmethod
    def parse(*, text: str, num_skills: int, param_dims: list[int]) -> SkillChoice:
        """The reply as a decision against the applicable set it was offered.

        `param_dims` is indexed by the same `skill_index` the agent returns, so the width
        check is against the skill actually chosen rather than against some global
        constant."""
        payload = DecisionParser.extract_object(text=text)
        if "skill_index" not in payload:
            raise DecisionParseError(f"reply has no 'skill_index' key (keys: {sorted(payload)})")
        raw_index = payload["skill_index"]
        if isinstance(raw_index, bool) or not isinstance(raw_index, int):
            raise DecisionParseError(f"skill_index is not an integer: {raw_index!r}")
        if not 0 <= raw_index < num_skills:
            raise DecisionParseError(
                f"skill_index {raw_index} is outside the applicable set (0..{num_skills - 1})"
            )
        params = DecisionParser.parse_params(
            raw=payload.get("params", []), expected_dim=param_dims[raw_index]
        )
        return SkillChoice(skill_index=raw_index, params=params)

    @staticmethod
    def extract_object(*, text: str) -> dict[str, Any]:
        """The JSON object in `text`, whether or not anything surrounds it.

        Scans for the LAST balanced `{...}` span rather than the first, because when the
        agent does narrate, the object it settles on comes after the reasoning, and an
        earlier brace is usually a hypothetical it went on to reject. Balanced-brace
        scanning rather than a regular expression: a parameter list is flat but a nested
        object is not, and a non-greedy regex silently truncates one."""
        if not text.strip():
            raise DecisionParseError("empty reply")
        for start, end in reversed(DecisionParser.brace_spans(text=text)):
            try:
                payload = json.loads(text[start:end])
            except json.JSONDecodeError:
                continue
            if isinstance(payload, dict):
                return payload
        raise DecisionParseError(f"no JSON object found in reply: {text[:200]!r}")

    @staticmethod
    def brace_spans(*, text: str) -> list[tuple[int, int]]:
        """Every balanced `{...}` span in `text`, as `(start, end)` half-open indices, in
        order of where each one opens. Braces inside a JSON string literal are skipped, so
        a reply that quotes a brace does not desynchronise the scan."""
        spans: list[tuple[int, int]] = []
        depth = 0
        start = 0
        in_string = False
        escaped = False
        for index, char in enumerate(text):
            if in_string:
                if escaped:
                    escaped = False
                elif char == "\\":
                    escaped = True
                elif char == '"':
                    in_string = False
                continue
            if char == '"':
                in_string = True
            elif char == "{":
                if depth == 0:
                    start = index
                depth += 1
            elif char == "}" and depth > 0:
                depth -= 1
                if depth == 0:
                    spans.append((start, index + 1))
        return spans

    @staticmethod
    def parse_params(*, raw: Any, expected_dim: int) -> tuple[float, ...]:
        """`expected_dim` finite floats, or a `DecisionParseError` naming what was wrong.

        A `param_dim == 0` skill takes an empty list, and `None` is accepted there as well
        as `[]`: the two are the same statement and rejecting one would be a parser
        preference rather than a real distinction."""
        if raw is None:
            raw = []
        if not isinstance(raw, list):
            raise DecisionParseError(f"params is not a list: {raw!r}")
        if len(raw) != expected_dim:
            raise DecisionParseError(
                f"params has {len(raw)} entries but the chosen skill takes {expected_dim}"
            )
        params: list[float] = []
        for entry in raw:
            if isinstance(entry, bool) or not isinstance(entry, (int, float)):
                raise DecisionParseError(f"param is not a number: {entry!r}")
            value = float(entry)
            if not math.isfinite(value):
                raise DecisionParseError(f"param is not finite: {entry!r}")
            params.append(value)
        return tuple(params)
