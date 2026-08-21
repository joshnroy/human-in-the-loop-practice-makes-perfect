import abc

import numpy as np
from pydantic import BaseModel, ConfigDict

from hitl_pmp.core.method.types import GroundSkill, LabeledAction, Skill
from hitl_pmp.core.problem.environment.environment import Environment
from hitl_pmp.core.problem.environment.types import Action, Object, State, Type
from hitl_pmp.core.problem.tasks.types import Goal, Predicate

# The one name `_EesEpisode.step` checks for to intercept this ground skill before it
# would otherwise be dispatched through the normal controller/skill-execution path --
# the same interception `ASK_FOR_RESET_TASK_INITIAL_NAME` gets, for the same reason
# (this "skill" has no controller and must never reach execute_ground_skill). Defined
# here, in core/, rather than in methods/practice_makes_perfect/ alongside that
# constant, because a concrete SkillProvider (e.g. Tossing3DSkillProvider) is what
# names its returned GroundSkill -- and a SkillProvider must not import methods/
# (methods sits above environments in the layered contract, not below it). This is
# the one piece of the contract both sides need without either importing the other.
ASK_FOR_RESET_CUBE_BIN_ONLY_NAME = "ask_for_reset_cube_bin_only"


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

    def oracle_sampler_input(
        self, *, ground_skill: GroundSkill, state: State, params: np.ndarray
    ) -> list[float] | None:
        """Optional per-domain "oracle feature selection" for the learned sampler's
        classifier input -- predicators'
        `active_sampler_learning_feature_selection = "oracle"` branch of
        `utils.construct_active_sampler_input`.

        Returns the *full* sampler input row (including the leading `1.0` bias) when
        this domain hand-picks a curated feature vector for this ground skill, or
        `None` to fall back to the default `"all"` layout
        (`[1.0] + concat(state[obj] for obj) + params`) the caller builds itself.

        Non-abstract with a `None` default so a domain that does no oracle feature
        selection (e.g. Light Switch) needs no override and is left exactly as it was.
        A concrete provider that overrides this must build the row consistently for
        both a training observation and a candidate being scored -- i.e. it is a pure
        function of `(ground_skill, state, params)`."""
        return None

    def human_cube_bin_reset_skill(self) -> GroundSkill | None:
        """A domain-specific ground skill for `HumanCubeBinResetRequested`: offered
        to `EesMethod`'s planner as a mid-plan step, priced by `plan_to` itself
        (`--ask-for-reset-cube-bin-cost`), not here.

        Unlike `ask_for_reset_task_initial`/`ask_for_reset_random_task` (built
        generically from a domain's `objects`/`predicates`/`init_atoms`), this
        skill's effect -- "the objects a human could tidy up are back in place" --
        can only be written in terms of *this domain's own* predicates, so each
        domain's `SkillProvider` builds its own operator here.

        `GroundSkill.skill.name` must equal `ASK_FOR_RESET_CUBE_BIN_ONLY_NAME`
        exactly (`_EesEpisode.step` intercepts on it). `None` default: most domains
        have no robot/non-robot distinction to offer, and `plan_to` never adds the
        skill regardless of the cost flag when this returns `None`."""
        return None


class OraclePolicyProvider(BaseModel, abc.ABC):
    """A domain's privileged, hand-authored solver -- what `SkillOracleMethod` drives
    as its upper-bound baseline. Reads ground-truth state and returns the next
    LabeledAction toward the goal. Domain-specific by nature (it encodes how to solve
    that domain), so it is injected the same way `SkillProvider` is, rather than
    being reimplemented as a second Method subclass per domain."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @abc.abstractmethod
    def get_labeled_action(self, *, state: State, goal: Goal) -> LabeledAction:
        """Next privileged action toward `goal`. Goal-agnostic oracles (Light Switch,
        Ball-Ring drive toward a single fixed objective from state alone) ignore it;
        a goal-dependent oracle (Tossing Room, whose state can't distinguish
        throw-recycling from throw-trash) reads it to pick which item/bin/room."""
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
