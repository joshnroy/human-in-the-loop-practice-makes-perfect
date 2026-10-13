"""Agent-facing text for the ``--no-human-reset`` arm.

Every table entry is an exact substring of the text it targets, and ``remove`` raises
when one is missing: a prompt edit that strands an entry fails here, loudly, instead of
leaking a human skill, its price, or the practice-side hint into the arm that must not
see them. The default arm never passes through this module, so its renderings are the
raw files byte for byte.
"""

from typing import ClassVar

Replacements = tuple[tuple[str, str], ...]


class HumanResetText:
    TASK_PROMPT: ClassVar[Replacements] = (
        (
            "Use `env_client.py` for serialized robot commands and human requests.",
            "Use `env_client.py` for serialized robot commands.",
        ),
        (
            "Available human skills during practice: `reset_cube_far` samples a cube "
            "placement in the robot-side pickup area and a bin placement across the "
            "barrier; `reset_cube_and_bin_near` samples both on the robot side. Both "
            "reposition both objects, preserve the arm and gripper state, and normally "
            "preserve the base; the host may return the base to its initial pose if it "
            "blocks valid bin placements. The supplied interventions are deterministic "
            "host operations. Each has a separate cost function cₕ(e) and duration "
            "function dₕ(e), both equal to 1 initially.\n",
            "",
        ),
        (
            " Hₕ denotes dispatched invocations of human skill h; cₕ is its cost function "
            "and dₕ its counted-duration function.",
            "",
        ),
        ("$$\nC_H = \\sum_h \\sum_{e\\in H_h} c_h(e)\n$$\n", ""),
        ("C = C_R + w_H C_H\n", "C = C_R\n"),
        ("B = |R| + \\sum_h \\sum_{e\\in H_h} d_h(e)\n", "B = |R|\n"),
        (
            "$$\nc_h(e) \\equiv 1 \\quad \\forall h\n$$\n"
            "$$\nd_h(e) \\equiv 1 \\quad \\forall h\n$$\n"
            "$$\nw_H = 1\n$$\n",
            "",
        ),
        ("w_H weights human cost relative to robot cost. ", ""),
        (" Each dispatched human invocation counts once, including failure.", ""),
        (
            "Practice experience is B = robot control periods + Σdₕ(e), with dₕ(e) = 1 for "
            "each human skill.",
            "Practice experience is B = robot control periods.",
        ),
        ("robot/human cost tradeoff", "robot cost tradeoff"),
        (
            "(choosing a human reset, starting a robot controller execution",
            "(starting a robot controller execution",
        ),
        ("expected subsequent robot steps and human requests", "expected subsequent robot steps"),
        (
            "Human assistance and model calls are unavailable during deployment.",
            "Model calls are unavailable during deployment.",
        ),
        (
            "\nhint: in order to minimize human cost, consider practicing tosses on the "
            "robot side and then transferring it zero shot to the far side. when practicing "
            "tosses on the robot side, minimize human cost by picking up the cube after "
            "each toss (either from inside or outside the bin). parameterize your tossing "
            "controller in such a way that it will zero-shot transfer from same side to "
            "far side\n",
            "",
        ),
    )
    CONTINUATION: ClassVar[Replacements] = (
        ("expected accumulated robot and human cost", "expected accumulated robot cost"),
    )
    FRAGILE: ClassVar[Replacements] = (
        (
            "All robot steps and human requests retain their configured costs.",
            "All robot steps retain their configured costs.",
        ),
        ("Initial placement and human placement are exempt.", "Initial placement is exempt."),
        ("alongside robot and human costs", "alongside robot costs"),
    )
    BUDGET: ClassVar[Replacements] = (
        (
            "one per robot control period plus each human skill's configured duration.",
            "one per robot control period.",
        ),
    )
    SYSTEM_PROMPT: ClassVar[Replacements] = (
        (
            "Robot control steps and human-skill invocations incur the specified costs. "
            "At deployment the frozen controller runs autonomously without human assistance "
            "or model calls.",
            "Robot control steps incur the specified costs. At deployment the frozen "
            "controller runs autonomously without model calls.",
        ),
    )
    ROBOT_API: ClassVar[Replacements] = (
        (
            "`request_help(intervention_id)` invokes the named human skill and returns the "
            "resulting observation, outcome, and accounting.\n",
            "",
        ),
        ("env.request_help(intervention_id)\n", ""),
    )
    ENV_CLIENT: ClassVar[Replacements] = (
        (
            "    def request_help(self, intervention_id: Any) -> Any:  # noqa: PLR0917 -- "
            "supplied RoboCode client/interface contract\n"
            '        return self.request("request_help", intervention_id=intervention_id)\n'
            "\n",
            "",
        ),
    )

    @staticmethod
    def remove(*, text: str, replacements: Replacements) -> str:
        for old, new in replacements:
            if text.count(old) != 1:
                raise ValueError(f"Expected exactly one occurrence of {old[:60]!r}")
            text = text.replace(old, new)
        return text
