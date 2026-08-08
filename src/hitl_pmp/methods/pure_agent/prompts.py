import enum
from typing import Any

from hitl_pmp.core.method.skill_provider import SkillProvider
from hitl_pmp.core.method.types import Skill, SkillPracticeTally
from hitl_pmp.methods.pure_agent.observation import ObservationBuilder

# The contract half of the opening prompt: what the agent is being asked to do and the
# exact shape of every answer. Identical on both arms and in both phases -- the arms
# differ only in whether a natural-language account of the domain is appended, and the
# phases only in what feedback is allowed, so anything else differing would confound
# either comparison.
#
# The schema is carried over verbatim from the closed authoring stack's own `_CONTRACT`
# (PRs #163-#177), minus everything about writing a file: the observation shape, the
# `{"skill_index": ..., "params": [...]}` reply and the "index the applicable set"
# guarantee were all already settled there and are not worth re-deriving.
CONTRACT = """\
You ARE the control policy for the robotics domain described below. At every step of
every episode you will be sent one observation, and you must reply with one line of JSON
and nothing else:

{"skill_index": <int>, "params": [<float>, ...]}

`skill_index` indexes `observation["skills"]`, the ground skills whose preconditions hold
RIGHT NOW -- so any index in range names a legal action, and there is no way to select
something inapplicable. `params` must have exactly `param_dim` finite floats for the
skill you chose; a skill with `param_dim` 0 takes an empty list.

Reply with the JSON object only. No prose, no code fences, no explanation. Do not use
tools. Do not try to run the environment; it does not exist where you are.

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

WHAT_IS_UNKNOWN = """\
What a skill's continuous parameters MEAN, and what makes an action succeed, are NOT
given to you. The predicates and effects below say what a skill is supposed to achieve,
not how to achieve it. Where a skill takes parameters, finding the relationship between
the observable state and the parameter values that work is the actual problem.
"""

# What the agent is told about the two phases. Stated to it explicitly rather than left
# implicit, because the asymmetry is real and an agent that does not know which phase it
# is in would reasonably keep experimenting during evaluation.
PRACTICE_BRIEF = """\
You are in a PRACTICE period. Nothing here is scored. After each action you will be told
whether it achieved that skill's own declared add effects, and the running totals for
every skill. Use the period to find out how this world works -- deliberately trying a
parameter you expect to be wrong is a reasonable thing to do here and costs you nothing.
"""

EVALUATION_BRIEF = """\
You are being EVALUATED on a held-out task. Act to reach the goal. You will receive no
feedback of any kind between steps beyond the next observation, and nothing you do here
will be reported back to you or to any later episode. Exploit what you already know.
"""

DIGEST_REQUEST = """\
The practice period is over. Write down, for your own future use, what you have learned
about this domain: what the parameters mean, what values worked and did not, which
sequences of skills make progress toward which goals, and anything else you would want to
know if you woke up in this world with no memory of this conversation.

You will be given this note -- and nothing else from this conversation -- at the start of
every episode from now on. Write it for that reader. Be concrete and quantitative: an
equation or a table of values you have actually observed is worth more than advice. Reply
with the note itself and nothing else.
"""


class PromptArm(str, enum.Enum):
    """Which of the two prompt arms to run.

    `(str, Enum)` so argparse can offer the members directly as `choices`, so a member
    compares equal to its own wire string, and so the chosen value lands in
    `config_snapshot.json` as a readable word.

    The arms exist because `prpl-agent-utils`' own worked example flags its own prompt --
    *"The prompt names the environment, which is a large hint"* -- and that hint is not
    separable from the result unless both are run."""

    # The symbolic layer and nothing else: lifted skills, predicates, object types, and
    # the objects to ground over. Everything here comes out of `SkillProvider`, so this
    # arm knows exactly what a domain-agnostic planning Method knows and not one word
    # more. This is the arm with no hint in it.
    MINIMAL = "minimal"
    # The same, plus a natural-language account of the domain supplied by the operator
    # (`--pure-agent-domain-description`). It tells the agent what the world *is*, which
    # is knowledge no planning Method has and which the agent could not derive from the
    # symbols alone.
    DESCRIBED = "described"

    def __str__(self) -> str:
        return self.value


class PromptBuilder:
    """Renders every prompt sent to the agent. A static-method container, never
    instantiated, same as every other business-logic class in this project.

    **Domain-agnostic**: everything domain-specific is read off the injected
    `SkillProvider`, which is the same interface `RandomSkillsMethod` acts through, so
    this file contains no `isinstance` dispatch and no environment import and the prompts
    are correct on any `--env` without being touched.

    **The firewall lives in this file as much as in the method.** `evaluation_opening`
    takes no outcome, no tally and no metric; there is no parameter it could be handed one
    through. `practice_step` is the only prompt that carries an outcome, and its caller is
    the only caller on the practice side. A leak would have to add an argument here, which
    is a visible change to a signature rather than a line buried in a policy."""

    @staticmethod
    def practice_opening(
        *,
        skill_provider: SkillProvider,
        arm: PromptArm,
        domain_description: str,
        digest: str,
    ) -> str:
        """The first prompt of a practice period, before any observation.

        Sent once per period rather than once per step, because the conversation carries
        it forward -- and the period is reset at its boundary precisely so that this stays
        the head of a bounded conversation instead of the head of a run-length one."""
        return "\n".join([
            CONTRACT,
            PromptBuilder.symbolic_layer(skill_provider=skill_provider),
            *PromptBuilder.description_section(arm=arm, domain_description=domain_description),
            WHAT_IS_UNKNOWN,
            *PromptBuilder.digest_section(digest=digest),
            PRACTICE_BRIEF,
            "Acknowledge with the single word READY. The first observation follows.",
        ])

    @staticmethod
    def evaluation_opening(
        *,
        skill_provider: SkillProvider,
        arm: PromptArm,
        domain_description: str,
        digest: str,
    ) -> str:
        """The first prompt of one evaluation episode.

        Identical to `practice_opening` except for the brief, and **deliberately takes the
        same arguments**: the same domain, the same arm, the same digest. `digest` is the
        one and only channel by which anything learned during practice reaches evaluation,
        and it is produced from the practice conversation alone. There is no parameter here
        for an outcome, a tally or a score, and that absence is the firewall."""
        return "\n".join([
            CONTRACT,
            PromptBuilder.symbolic_layer(skill_provider=skill_provider),
            *PromptBuilder.description_section(arm=arm, domain_description=domain_description),
            WHAT_IS_UNKNOWN,
            *PromptBuilder.digest_section(digest=digest),
            EVALUATION_BRIEF,
            "Acknowledge with the single word READY. The first observation follows.",
        ])

    @staticmethod
    def description_section(*, arm: PromptArm, domain_description: str) -> list[str]:
        if arm is not PromptArm.DESCRIBED:
            return []
        return [f"## What this domain is\n\n{domain_description.strip()}\n"]

    @staticmethod
    def digest_section(*, digest: str) -> list[str]:
        """The note the agent wrote for itself at the end of the last practice period.

        Empty before any practice has happened, which is what makes the first evaluation
        sweep a genuine untrained baseline rather than one carrying a hand-written hint."""
        if not digest.strip():
            return []
        return [
            "## Your own notes from earlier practice in this world\n\n"
            f"{digest.strip()}\n\n"
            "These are your notes and may be wrong. Trust an observation over them.\n"
        ]

    @staticmethod
    def evaluation_step(*, observation: dict[str, Any]) -> str:
        """One evaluation decision point. The observation and nothing else.

        No outcome, no tally, no step counter, no remaining-horizon hint: every one of
        those is something the harness measured, and handing any of them over during
        evaluation is training on the test set."""
        return f"observation = {ObservationBuilder.render(observation=observation)}"

    @staticmethod
    def practice_step(
        *,
        observation: dict[str, Any],
        previous_outcome: str | None,
        practice_outcomes: dict[str, SkillPracticeTally],
    ) -> str:
        """One practice decision point: what happened to the last action, the running
        per-skill tallies, and the new observation.

        **Counts, never rates.** Every number here is `x/y`, because that is this project's
        standing rule and because it is the rule that matters most in a prompt: an agent
        told "83% success" cannot tell 5/6 from 830/1000, and will change a parameter it
        should have kept.

        `previous_outcome` is `None` on the first step of a period, where there is no
        previous action to report."""
        lines: list[str] = []
        if previous_outcome is not None:
            lines += [previous_outcome, ""]
        if practice_outcomes:
            lines.append(
                "Cumulatively over practice so far, each skill was executed this many "
                "times, and this many of those executions achieved the skill's own "
                "declared add effects:"
            )
            lines.append("")
            for name in sorted(practice_outcomes):
                tally = practice_outcomes[name]
                lines.append(f"  {name}: {tally.num_successes}/{tally.num_attempts}")
            lines.append("")
        lines.append(f"observation = {ObservationBuilder.render(observation=observation)}")
        return "\n".join(lines)

    @staticmethod
    def outcome_line(*, skill_label: str, achieved_add_effects: bool) -> str:
        """How the previous practice action is reported back.

        The add-effect check, not "did you solve the task": it is the same predicate
        `EesMethod` and `RandomSkillsMethod` score by, so this arm's practice numbers and
        theirs mean the same thing and can go on one chart."""
        verdict = "DID" if achieved_add_effects else "did NOT"
        return f"Your last action was {skill_label}. It {verdict} achieve its declared add effects."

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
