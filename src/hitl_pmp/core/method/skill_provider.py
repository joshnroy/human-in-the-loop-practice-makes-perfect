import abc

import numpy as np
from pydantic import BaseModel, ConfigDict

from hitl_pmp.core.method.types import GroundSkill, LabeledAction, Skill
from hitl_pmp.core.problem.environment.environment import Environment
from hitl_pmp.core.problem.environment.types import Action, Object, State, Type
from hitl_pmp.core.problem.tasks.types import Predicate


class SkillProvider(BaseModel, abc.ABC):
    """The one per-domain object a domain-agnostic learning/baseline Method needs to
    act: the domain's lifted skills, its predicates/types/objects (for symbolic
    grounding and planning), and the two functions that turn a chosen GroundSkill
    into a raw environment Action -- `sample_params` (a state-independent draw of a
    skill's continuous parameters, so a learned sampler can generate many candidates
    without the state) and `compute_action` (which reads the state to realize those
    parameters as an actual [.. ] action vector).

    This is the seam that lets `EesMethod`/`RandomSkillsMethod` be written once and
    run on any domain: previously they imported `LightSwitchSkills`/
    `LightSwitchEnvironment` directly and dispatched on `isinstance(self.env, ...)`.
    A concrete provider (e.g. `LightSwitchSkillProvider`, `BallRingSkillProvider`)
    lives under `environments/<domain>/` and is constructed by that domain's
    `cli.py` composition root, then injected into whichever Method is being driven.

    A pydantic `BaseModel, abc.ABC` (like `Environment`/`Tasks`): a concrete provider
    holds the one `Environment` instance it reads structural config from (e.g.
    Ball-Ring's table count, to enumerate `objects()`), as a constructor field."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @abc.abstractmethod
    def skills(self) -> tuple[Skill, ...]:
        """The domain's lifted skills, in a fixed order (the order a random baseline
        and the planner both enumerate)."""
        raise NotImplementedError

    @abc.abstractmethod
    def predicates(self) -> tuple[Predicate, ...]:
        raise NotImplementedError

    @abc.abstractmethod
    def types(self) -> tuple[Type, ...]:
        raise NotImplementedError

    @abc.abstractmethod
    def objects(self) -> tuple[Object, ...]:
        """Every object the symbolic layer grounds over -- fixed for a given
        Environment config (Light Switch's robot/light/cells; Ball-Ring's robot/
        ball/cup and its deterministic ring of tables)."""
        raise NotImplementedError

    @abc.abstractmethod
    def sample_params(self, *, ground_skill: GroundSkill, rng: np.random.Generator) -> np.ndarray:
        """A state-independent draw of `ground_skill.skill.param_dim` continuous
        parameters. State-independent on purpose: the learned sampler generates many
        candidates from this, then picks among them using the current state's
        features (see `EesMethod.execute_ground_skill`)."""
        raise NotImplementedError

    @abc.abstractmethod
    def compute_action(
        self, *, ground_skill: GroundSkill, params: np.ndarray, state: State
    ) -> Action:
        """Realize a chosen (ground skill, continuous params) as a raw Action,
        reading whatever the state provides (object positions, etc.)."""
        raise NotImplementedError


class OraclePolicyProvider(BaseModel, abc.ABC):
    """A domain's privileged, hand-authored solver -- what `SkillOracleMethod` drives
    as its upper-bound baseline. Reads ground-truth state and returns the next
    LabeledAction toward the goal. Domain-specific by nature (it encodes how to solve
    that domain), so it is injected the same way `SkillProvider` is, rather than
    being reimplemented as a second Method subclass per domain."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @abc.abstractmethod
    def get_labeled_action(self, *, state: State) -> LabeledAction:
        raise NotImplementedError


class DomainContext(BaseModel):
    """The bundle a domain's `cli.py` composition root builds and hands to a
    method-CLI's `method_factory`: the concrete Environment plus the domain's
    SkillProvider and OraclePolicyProvider. Lets a method-CLI construct its Method
    from a single argument regardless of which `--env` was selected, so the method
    side never imports a specific environment."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    env: Environment
    skill_provider: SkillProvider
    oracle: OraclePolicyProvider
