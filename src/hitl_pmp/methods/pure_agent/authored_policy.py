import math
import types
from collections.abc import Callable
from typing import Any

from pydantic import BaseModel, PrivateAttr

from hitl_pmp.methods.pure_agent.types import SkillChoice


class AuthoredPolicy(BaseModel):
    """One round's `policy.py`, compiled and callable.

    **This executes agent-written code in this process.** The sandbox protects the
    *authoring* step -- the agent runs in a container whose only writable host path is
    its own directory -- and it does not protect this step at all; the notebook this
    method follows flags the same gap in its own text. Closing it would mean evaluating
    inside the container too and passing only a score back, which is a larger change than
    this baseline needs and is recorded here as the known limitation it is. Replaying a
    transcript someone else authored runs their agent's code on your machine: read the
    `policy.py` in the artifact before replaying one you did not produce.

    Construction is the load step and can fail (`AuthoredPolicyError`); calling it can
    also fail, and `choose` converts every such failure into a `None` rather than letting
    it escape. Both are reported back to the agent in the next round's prompt, which is
    the only way it ever finds out."""

    model_config = {"frozen": True}

    source: str
    round_index: int

    _policy: Callable[[dict[str, Any]], Any] = PrivateAttr()
    # The FIRST call-time failure only -- see _record_failure.
    _failure: str | None = PrivateAttr(default=None)

    def model_post_init(self, __context: object) -> None:
        module = types.ModuleType(f"hitl_pmp_authored_policy_{self.round_index}")
        filename = f"<pure-agent policy.py, round {self.round_index}>"
        try:
            code = compile(self.source, filename, "exec")
        except SyntaxError as exc:
            raise AuthoredPolicyError(f"{type(exc).__name__}: {exc}") from exc
        try:
            exec(code, module.__dict__)  # noqa: S102 -- see the class docstring
        except Exception as exc:
            raise AuthoredPolicyError(f"{type(exc).__name__}: {exc}") from exc
        policy = getattr(module, "policy", None)
        if not callable(policy):
            raise AuthoredPolicyError(
                "policy.py defines no callable named `policy`. It must define exactly "
                "`def policy(observation): ...`."
            )
        self._policy = policy

    def choose(self, *, observation: dict[str, Any]) -> SkillChoice | None:
        """The policy's decision for this observation, or `None` if it did not make a
        usable one.

        `None` covers every way agent-authored code can misbehave at call time: raising,
        returning the wrong shape, naming a skill that is not in the applicable list, and
        returning a parameter vector of the wrong width or with a non-finite entry. The
        caller turns each into a no-op and counts it.

        **Every failure is caught here rather than validated by the caller**, because the
        caller cannot distinguish "the agent's code raised" from "the harness has a bug"
        once an exception is in flight, and swallowing the second would be much worse than
        reporting the first. A bare `except Exception` is correct at exactly one boundary,
        and this is it: untrusted code, called in-process, whose every failure mode is the
        experiment's data rather than the harness's."""
        try:
            returned = self._policy(observation)
        except Exception as exc:
            self._record_failure(reason=f"{type(exc).__name__}: {exc}")
            return None
        return self._validate(returned=returned, observation=observation)

    def _validate(self, *, returned: Any, observation: dict[str, Any]) -> SkillChoice | None:
        if not isinstance(returned, dict):
            self._record_failure(
                reason=f"policy returned {type(returned).__name__}, expected a dict with "
                "keys 'skill_index' and 'params'"
            )
            return None
        try:
            skill_index = int(returned["skill_index"])
            params = tuple(float(value) for value in returned.get("params", ()))
        except (KeyError, TypeError, ValueError) as exc:
            self._record_failure(reason=f"{type(exc).__name__}: {exc}")
            return None
        skills = observation["skills"]
        if not 0 <= skill_index < len(skills):
            self._record_failure(
                reason=f"skill_index {skill_index} is outside the {len(skills)} applicable "
                "skills it indexes"
            )
            return None
        expected_dim = int(skills[skill_index]["param_dim"])
        if len(params) != expected_dim:
            self._record_failure(
                reason=f"skill {skills[skill_index]['name']!r} takes {expected_dim} "
                f"parameters, got {len(params)}"
            )
            return None
        if not all(math.isfinite(value) for value in params):
            self._record_failure(reason=f"non-finite parameter in {list(params)}")
            return None
        return SkillChoice(skill_index=skill_index, params=params)

    def _record_failure(self, *, reason: str) -> None:
        """Remember the FIRST failure only.

        The first one is the one the agent caused; every later one in the same round is
        very likely the same bug hit again, and a sweep over 30 test tasks would otherwise
        overwrite it 30 times with a less informative instance of itself."""
        if self._failure is None:
            self._failure = reason

    def failure(self) -> str | None:
        """The first call-time failure this policy produced, or `None` if it has always
        returned a usable decision. Quoted back to the agent in the next round's prompt."""
        return self._failure


class AuthoredPolicyError(Exception):
    """The authored `policy.py` could not be turned into a callable at all -- it does not
    compile, it raised while being imported, or it defines no `policy`.

    Distinct from a call-time failure (`AuthoredPolicy.failure`) only in when it is
    found; both end in the same two places, a no-op action and the next round's prompt."""
