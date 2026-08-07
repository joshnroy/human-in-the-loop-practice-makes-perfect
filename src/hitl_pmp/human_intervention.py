import enum


class HumanResetTarget(str, enum.Enum):
    """WHAT configuration a human is asked to restore when a `Method` asks for help.

    **This module is the human's half, and only the human's half.** *When* to ask is
    the Method's business and lives in `methods/help_seeking.py`; the harness owns only
    the mechanism -- invoke the `HumanOracle`, price the command, bank the cost, carry
    on. An earlier version of this file also held the trigger and a policy object that
    watched the state stream from inside `PracticeLoop`, which is the external monitor
    Josh rejected: "part of 'when the agent asks for help' is part of the method --
    that's not a baseline", "we don't want this in the harness".

    The import layering agrees, and that is why the split lands here rather than being
    a matter of taste. `hitl_pmp.methods` sits *above* `hitl_pmp.method_runner`, so
    `MethodRunner` cannot import `methods.help_seeking` and could not build a trigger
    policy even if it wanted to. What it can still do is read this flag, because what
    the human does is a property of the human.

    The two axes stay orthogonal: the arms this ladder runs are points in the product
    of `--ask-for-help` and this flag, not separate code paths.

    A `(str, Enum)` matching this project's other flag enums (`PracticeResetPolicy`,
    `TossingRoomGoalType`) so argparse can offer the members directly as `choices`, a
    member compares equal to its own wire string, and the chosen value lands in
    `config_snapshot.json` as a readable word rather than an integer."""

    # The initial state of the train task the current practice period was given -- "put
    # it back where this period started".
    TASK_INITIAL = "task-initial"
    # The initial state of a *freshly sampled* train task -- "put it somewhere else in
    # the task distribution". Deliberately expressed through `Tasks.sample_train_task`
    # rather than by perturbing the state directly, because that is the only
    # domain-agnostic notion of "a random state" available: nothing here knows which
    # feature vectors a domain considers reachable, and a uniformly perturbed State
    # would usually not be one. It does advance the train-task stream, which is a real
    # and intended difference between the two targets, not a leak.
    RANDOM = "random"

    def __str__(self) -> str:
        return self.value
