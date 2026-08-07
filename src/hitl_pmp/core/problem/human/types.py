from pydantic import BaseModel

from hitl_pmp.core.problem.environment.types import State
from hitl_pmp.core.problem.tasks.types import Goal


class CommandStartStateDescription(BaseModel):
    # TODO: figure out what this should actually contain. The design doc notes
    # humans can't really operate on raw states -- v3 proposes natural-language
    # and/or pictorial descriptions instead of a raw State. For now this just
    # wraps State as a placeholder.
    state: State


class CommandGoalDescription(BaseModel):
    # TODO: see CommandStartStateDescription -- same open question, even though this
    # one already uses the same symbolic Goal as Task.goal rather than a raw State.
    goal: Goal
    # The exact configuration the human is asked to restore, when the command is a
    # **reset** rather than a goal to bring about. None (the default) is the original
    # shape: "make this goal true, however you like".
    #
    # This is not a retreat from the symbolic goal above, it is a second kind of command.
    # A physical human reset -- picking the robot up and carrying it back -- IS a state
    # teleport; there is no goal being achieved, and asking for one would misdescribe
    # what happened. And a domain-agnostic HumanOracle cannot go the other way: a Goal is
    # a frozenset of GroundAtoms whose truth is an opaque `holds` callable, so nothing
    # outside a domain can synthesise a State that satisfies one. Without this field the
    # v0 oracle (humans/oracle.py) is not implementable at all, which is why it was added
    # alongside it rather than speculatively.
    #
    # `goal` stays required and stays meaningful for a reset: it is what the robot was
    # pursuing when it got stuck, which is exactly what a later, capability-aware human
    # (v1+) would price the rescue against -- "put it somewhere it can still finish this"
    # is a different request from "put it anywhere".
    target_state: State | None = None


Cost = float
