import enum

from hitl_pmp.core.method.skill_provider import SkillProvider
from hitl_pmp.core.method.types import Skill, SkillPracticeTally
from hitl_pmp.methods.pure_agent.types import PracticeTransition

# The contract half of every prompt: what the agent must write and what it will be
# called with. Identical on both arms -- the arms differ only in whether a
# natural-language account of the domain is appended, so anything else differing
# between them would confound the comparison.
_CONTRACT = """\
Write a file `policy.py` in the current directory containing a control policy for the
robotics domain described below.

The file must define exactly this function:

    def policy(observation):
        # See the observation schema below.
        # Return {"skill_index": <int>, "params": [<float>, ...]}.

`skill_index` indexes `observation["skills"]`, the ground skills whose preconditions
hold RIGHT NOW -- so any index in range names a legal action, and there is no way to
select something inapplicable. `params` must have exactly `param_dim` finite floats for
the skill you chose; a skill with `param_dim` 0 takes an empty list.

Only the standard library and `numpy` may be imported. The function must be
deterministic and fast: it is called once per environment step, thousands of times per
evaluation. Do not try to run the environment; it does not exist in this sandbox. The
policy is evaluated outside the sandbox and you will be told how it did.

The observation is a plain JSON-shaped dict:

    {
      "goal":    ["Predicate(obj, obj)", ...],   # atoms that must ALL hold to solve
      "objects": [{"name": str, "type": str, "features": {name: float}}, ...],
      "atoms":   ["Predicate(obj, obj)", ...],   # every atom true right now
      "skills":  [{"index": int, "name": str, "objects": [str, ...],
                   "param_dim": int}, ...]
    }

`goal` and `atoms` use the same rendering, so you can test goal satisfaction by string
membership. `objects` carries the raw continuous state; `atoms` is its symbolic
abstraction. Both are always present and consistent with each other.
"""

_WHAT_IS_UNKNOWN = """\
What a skill's continuous parameters MEAN, and what makes an action succeed, are NOT
given to you. The predicates and effects below say what a skill is supposed to achieve,
not how to achieve it. Where a skill takes parameters, finding the relationship between
the observable state and the parameter values that work is the actual problem -- reason
about it from the feedback you are given between rounds.
"""


class PromptArm(str, enum.Enum):
    """Which of the two prompt arms to run.

    `(str, Enum)` for the same three reasons `PracticeResetPolicy` is: argparse can offer
    the members directly as `choices`, a member compares equal to its own wire string, and
    the chosen value lands in `config_snapshot.json` as a readable word.

    The arms exist because the notebook flags its own prompt -- *"The prompt names the
    environment, which is a large hint"* -- and that hint is not separable from the result
    unless both are run. **They are not a measurement of what the hint is worth at one
    seed**; at one seed they establish only that the plumbing carries both."""

    # The symbolic layer and nothing else: lifted skills, predicates, object types, and
    # the objects to ground over. Everything here comes out of `SkillProvider`, so this
    # arm knows exactly what a domain-agnostic planning Method knows and not one word
    # more. This is the arm with no hint in it.
    MINIMAL = "minimal"
    # The same, plus a natural-language account of the domain supplied by the operator
    # (`--pure-agent-domain-description`). The analogue of the notebook naming
    # `Pendulum-v1`: it tells the agent what the world *is*, which is knowledge no
    # planning Method has and which the agent could not derive from the symbols alone.
    DESCRIBED = "described"

    def __str__(self) -> str:
        return self.value


class FeedbackArm(str, enum.Enum):
    """Which of the two *feedback* baselines to run. Orthogonal to `PromptArm`, which
    varies what the agent is told about the domain up front; this varies what it is told
    about its own behaviour between rounds.

    `(str, Enum)` for the same three reasons `PromptArm` is.

    **"Zero-shot" here means no worked examples, not no feedback.** Both arms get the
    aggregate `x/y` tallies -- that is what makes the axis clean, since a difference
    between them is then a difference in examples and nothing else. The names are the
    in-context-learning ones because that is exactly the distinction: the in-context arm's
    prompt carries demonstrations of `(observation, action, outcome)`, and the zero-shot
    arm's carries none. Do not read `ZERO_SHOT` as "round 0 only"."""

    # What this method shipped with, unchanged: per-lifted-skill `num_successes/
    # num_attempts` and the practice-goal count. The control arm.
    ZERO_SHOT = "zero-shot"
    # The same, plus a `model_dump_json()` dump of the parameterized practice transitions
    # -- what the policy saw, what parameters it chose, and whether they worked.
    IN_CONTEXT = "in-context"

    def __str__(self) -> str:
        return self.value


# Deliberately a cap on RECORDS rather than on characters or tokens. A record is the unit
# an agent reasons over, and half a record is worse than none; a byte budget would also
# vary the count between domains and make two runs incomparable for a reason that has
# nothing to do with either.
#
# **40 rather than "one period's worth", and the reason is identifiability.** A single
# period's records come from a single policy, so they can carry no variation in the
# parameter that policy chose. Rendered against the 2026-08-07 pilot's own round-0 policy,
# all 19 Tossing Room throws sit at `force=2.0` with `achieved_add_effects=False`; only the
# item weight varies. That dump says "2.0 is wrong" and cannot say what is right -- a
# relation between an observable and a working parameter needs two different parameters
# tried, and those only ever come from two different rounds. So the window is sized to span
# more than one period (~20 throws each on that pilot) and keep the previous policy's
# attempts alive beside the current one's.
#
# 40 at ~2.9 KB per Tossing Room observation is ~116 KB, comfortably inside a context
# window. Kept finite because a run with many cycles would otherwise grow the prompt
# without bound.
DEFAULT_MAX_DUMPED_TRANSITIONS = 40


class PromptBuilder:
    """Renders the prompts sent to the agent. A static-method container, never
    instantiated, same as every other business-logic class in this project.

    **Domain-agnostic**: everything domain-specific is read off the injected
    `SkillProvider`, which is the same interface `RandomSkillsMethod` acts through, so
    this file contains no `isinstance` dispatch and no environment import and the prompts
    are correct on any `--env` without being touched."""

    @staticmethod
    def initial(*, skill_provider: SkillProvider, arm: PromptArm, domain_description: str) -> str:
        sections = [_CONTRACT, PromptBuilder.symbolic_layer(skill_provider=skill_provider)]
        if arm is PromptArm.DESCRIBED:
            sections.append(f"## What this domain is\n\n{domain_description.strip()}\n")
        sections.append(_WHAT_IS_UNKNOWN)
        return "\n".join(sections)

    @staticmethod
    def revision(
        *,
        skill_provider: SkillProvider,
        arm: PromptArm,
        domain_description: str,
        practice_outcomes: dict[str, SkillPracticeTally],
        num_practice_goals_reached: int,
        num_practice_periods: int,
        previous_error: str | None,
        feedback_arm: FeedbackArm = FeedbackArm.ZERO_SHOT,
        practice_transitions: tuple[PracticeTransition, ...] = (),
        max_dumped_transitions: int = DEFAULT_MAX_DUMPED_TRANSITIONS,
    ) -> str:
        """The feedback prompt: how the last round's policy did, and what to do next.

        **Counts, never rates.** Every number here is `x/y`, because that is this
        project's standing rule and because it is the rule that matters most in a prompt:
        an agent told "83% success" cannot tell 5/6 from 830/1000, and will revise a
        policy it should have kept.

        **Practice numbers only, never evaluation ones.** A `Method` is not shown its own
        evaluation results -- `Metrics` owns those -- and that is what makes training on
        the test set structurally impossible here rather than merely discouraged. The
        agent is told what happened during practice on training tasks, which is exactly
        the notebook's "evaluate on the training seeds and hand the score back".

        **The ERROR branch restates the whole task; the feedback branch does not.** Both
        rely on the CLI's `--continue` to carry the conversation, and that reliance is
        safe only when the previous query *finished*. It does not always: a query stopped
        by its own `--max-budget-usd` cap can be cut off mid-turn, and the resumed session
        then carries no usable memory of what was asked. Measured on the described arm of
        the 2026-08-07 Tossing Room pilot -- round 0 was cut off before writing anything,
        and rounds 1 and 2 received only "Your policy could not be evaluated: policy.py
        was not created. Fix `policy.py`." The agent searched the sandbox, found no file
        and no task specification, and correctly reported that it had nothing to go on.
        All three rounds produced no policy, at $1.9544, and the arm measured nothing.

        A cut-off is *precisely* the case that reaches this branch, so this branch is the
        one that cannot assume context. The feedback branch is only reached after a round
        that ran to completion and left a loadable file, where `--continue` has been
        observed to work, so it is left short deliberately rather than by omission."""
        if previous_error is not None:
            return (
                f"Your previous attempt did not produce a usable policy: {previous_error}\n\n"
                "Start again from the full task below. Do not assume anything from "
                "earlier in this conversation is still available to you -- write "
                "`policy.py` from scratch if it does not already exist.\n\n"
                + PromptBuilder.initial(
                    skill_provider=skill_provider, arm=arm, domain_description=domain_description
                )
            )
        lines = [
            f"Your policy reached the practice task's goal in "
            f"{num_practice_goals_reached}/{num_practice_periods} of the practice "
            "periods so far.",
            "",
            "Cumulatively over practice, each skill was executed this many times, and "
            "this many of those executions achieved the skill's own declared add "
            "effects:",
            "",
        ]
        if practice_outcomes:
            for name in sorted(practice_outcomes):
                tally = practice_outcomes[name]
                lines.append(f"  {name}: {tally.num_successes}/{tally.num_attempts}")
        else:
            lines.append("  (no skill was executed during practice)")
        lines += [
            "",
            "A skill with a low ratio is one whose parameters are wrong, or one being "
            "attempted in states where it cannot work. A skill never executed at all is "
            "one your policy never selected -- which may itself be the problem.",
            "",
        ]
        if feedback_arm is FeedbackArm.IN_CONTEXT:
            lines += PromptBuilder.transition_dump(
                practice_transitions=practice_transitions,
                max_dumped_transitions=max_dumped_transitions,
            )
        lines += [
            "Revise `policy.py` to solve more tasks. Keep the same function signature "
            "and the same return shape.",
            "",
        ]
        return "\n".join(lines)

    @staticmethod
    def transition_dump(
        *,
        practice_transitions: tuple[PracticeTransition, ...],
        max_dumped_transitions: int,
    ) -> list[str]:
        """The in-context arm's worked examples: one `model_dump_json()` per line.

        **JSON rather than prose**, because the agent is being asked to do arithmetic on
        these and a schema it can `json.loads` beats one it has to read. One record per
        line so the dump stays greppable and a truncated tail is still parseable.

        **The observation is dumped whole**, not reduced to the features that happen to
        matter. Which feature matters is precisely what the agent is being asked to work
        out, and picking for it would smuggle in the domain knowledge the MINIMAL arm
        exists to withhold -- the comparison would then be against a hint, not a method.

        **Truncation is announced, never silent.** An agent told "these are your throws"
        reasons about the denominator it was shown; if that is 30 of 97 it must be told so,
        or every rate it computes is wrong. Reported `x/y`, like every count in this
        project. The most recent are kept: they are the ones the policy under revision
        produced, and older records describe code that no longer exists."""
        total = len(practice_transitions)
        if total == 0:
            return [
                "No parameterized skill was executed during practice, so there are no "
                "per-attempt records to show you. A skill your policy never selects "
                "produces no evidence about its parameters.",
                "",
            ]
        kept = practice_transitions[-max_dumped_transitions:]
        header = (
            f"Below are {len(kept)}/{total} of the individual parameterized skill "
            "executions from practice, one JSON record per line."
        )
        if len(kept) < total:
            header += (
                f" The other {total - len(kept)}/{total} are omitted to bound this "
                "prompt; these are the most recent ones."
            )
        return [
            header,
            "",
            "Each record has the `observation` your policy was called with, the "
            "`skill_index` and `params` it returned, and `achieved_add_effects`: whether "
            "the skill achieved its own declared add effects once executed. This is the "
            "evidence for what your parameters actually do -- the relationship between "
            "the observable state and the values that work is recoverable from it.",
            "",
            *(transition.model_dump_json() for transition in kept),
            "",
        ]

    @staticmethod
    def symbolic_layer(*, skill_provider: SkillProvider) -> str:
        """The domain's types, objects, predicates and lifted skills, as text.

        This is the whole of the MINIMAL arm's domain knowledge, and it is exactly what a
        planning Method gets: `SkillProvider.types/objects/predicates/skills`. Nothing is
        summarised or omitted -- an effect the agent is not shown is one it cannot plan
        against, and the point of the arm is that the two see the same thing."""
        types = "\n".join(
            f"  {object_type.name}({', '.join(object_type.feature_names)})"
            for object_type in skill_provider.types()
        )
        objects = "\n".join(f"  {obj.name}: {obj.type.name}" for obj in skill_provider.objects())
        predicates = "\n".join(
            f"  {predicate.name}({', '.join(t.name for t in predicate.types)})"
            for predicate in skill_provider.predicates()
        )
        skills = "\n\n".join(
            PromptBuilder.render_skill(skill=skill) for skill in skill_provider.skills()
        )
        return (
            "## Object types (name, then its continuous features)\n\n"
            f"{types}\n\n"
            "## Objects in this world\n\n"
            f"{objects}\n\n"
            "## Predicates\n\n"
            f"{predicates}\n\n"
            "## Skills\n\n"
            f"{skills}\n"
        )

    @staticmethod
    def render_skill(*, skill: Skill) -> str:
        """One lifted skill in a PDDL-ish shape. `ignore_effects` is rendered too, and
        spelled out rather than named: it is the field that makes a non-monotone skill
        expressible (a move that invalidates every reachability atom, not just its own),
        and a reader who skips it will plan sequences the world refuses."""
        parameters = ", ".join(f"?{p.name} - {p.type.name}" for p in skill.parameters)
        lines = [f"  {skill.name}({parameters})  [param_dim={skill.param_dim}]"]
        for label, atoms in (
            ("preconditions", skill.preconditions),
            ("add effects", skill.add_effects),
            ("delete effects", skill.delete_effects),
        ):
            rendered = sorted(
                f"{atom.predicate.name}({', '.join('?' + v.name for v in atom.variables)})"
                for atom in atoms
            )
            lines.append(f"    {label}: {', '.join(rendered) if rendered else '(none)'}")
        if skill.ignore_effects:
            names = ", ".join(sorted(predicate.name for predicate in skill.ignore_effects))
            lines.append(
                f"    after this skill runs, EVERY atom of these predicates becomes "
                f"false, whatever its objects: {names}"
            )
        return "\n".join(lines)
