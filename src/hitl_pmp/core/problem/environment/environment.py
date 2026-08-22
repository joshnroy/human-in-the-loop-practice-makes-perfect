import abc
from typing import ClassVar

import numpy as np
from gymnasium.spaces import Space
from pydantic import BaseModel

from .types import Action, State


class Environment(BaseModel, abc.ABC):
    """The real-world environment (or the real/ground-truth simulator standing in for
    it) -- a real, constructor-injected instance now (not a static-method container):
    grid_size-style per-run config and current_state are genuine per-instance state,
    not shared globally. It is **not** a reusable, stateless dynamics function that
    other code can call with a hypothetical state to explore "what if" -- a `Method`
    that needs to plan carries its own model for that; it must not borrow
    `Environment` to do it. `take_action(*, action)` advances `current_state` by one
    action via the domain's own underlying dynamics and returns the new state;
    `get_valid_actions()` reads from `current_state` too -- neither takes an explicit
    `state` argument, both operate on this instance's one real state.
    `get_current_state()`/`set_state()` are concrete (shared across every
    `Environment`, not reimplemented per domain) -- `set_state` is a *privileged
    external override* (used by a human, via `HumanOracle`/
    `Problem.execute_human_command`, to force a state -- distinct from
    `take_action`'s normal forward dynamics). `hard_reset()` resets to the initial
    state distribution but is only ever called by the harness before a run starts,
    never by the agent itself.

    Deliberately no reward function: this is a multi-task environment, and success
    is judged by goal-state reaching (Task.goal.is_satisfied), not a scalar reward --
    a fixed reward wouldn't make sense across tasks the agent invents for itself.

    action_space stays a ClassVar: it's a structural constant of the domain (same
    for every instance that will ever exist), not per-run configuration -- unlike
    current_state, nothing about it varies between two LightSwitchEnvironment
    instances constructed with different grid_size/tolerance values.
    """

    action_space: ClassVar[Space]
    current_state: State | None = None

    def get_current_state(self) -> State:
        assert self.current_state is not None, (
            "No current_state yet -- call hard_reset() or set_state() first."
        )
        return self.current_state

    @abc.abstractmethod
    def take_action(self, *, action: Action) -> State:
        """Advances current_state by one action, via the domain's own underlying
        dynamics (not via set_state, which is a privileged external override), and
        returns the new state."""
        raise NotImplementedError

    @abc.abstractmethod
    def get_valid_actions(self) -> list[Action]:
        raise NotImplementedError

    @abc.abstractmethod
    def noop_action(self) -> Action:
        """The action that means "do nothing" in *this* domain's action space.

        A `Method` sometimes has to act without wanting to: a planner that finds no
        plan still owes the harness an `Action` for this step, because ending the
        episode is `Problem.run_task_episode`'s job and not the policy's. That
        placeholder must be genuinely inert, or a method that failed to decide gets
        scored on whatever the placeholder happened to trigger.

        Only the environment can answer, which is why this lives here and not on
        `Method`. `np.zeros(action_space.shape)` is the obvious guess, and across the
        six domains here it is right for the wrong reason twice, wrong twice, and
        right by construction once:

        - **Ball-Ring**: slot 0 is binary and zero means *navigate*, so zeros is
          "navigate to (0, 0)" -- it really moves the robot.
        - **Tossing3D**: `pick_id == 0`, so zeros is a real `pick_shelf` at distance
          0.0 -- a whole arm trajectory in the simulator.
        - **the three Tossing Rooms**: `SKILL_PICKUP == 0`, so zeros *decodes as*
          `Pickup`, but its item-kind argument also rounds to 0, which names no kind,
          so nothing is written. Inert by coincidence of a second field, not by
          design -- and anything reading the skill id mislabels it.
        - **Light Switch**: both slots are deltas, so zeros is inert by construction.
          This is the exception that made the guess look safe.

        Abstract rather than a `np.zeros` default for exactly that reason: a default
        here would be a plausible-looking answer that is silently wrong for the next
        domain added, which is the defect this method exists to close. A few lines of
        boilerplate per domain is the price of never re-opening it.

        The contract is that the domain's own dynamics ignore it: it triggers no
        skill and changes nothing a `Predicate` can observe. It is *not* a promise
        that no bookkeeping moves -- a simulator-backed domain still counts the
        transition (Tossing3D advances `steps_taken`), and the harness still charges
        the step, which is correct: refusing to act is a choice that costs time.

        Two things implementations may rely on and callers must respect. It may read
        `current_state`, so it is only callable once the environment has one
        (Ball-Ring's is the robot's own position, which is the only inert action its
        bounded Box can express). And it must return an action within that domain's
        own `action_space` *bounds*, since a `Method` may only emit actions from it --
        bounds rather than literal `action_space.contains`, because every action in
        this repo is float64 while a Box may be float32, and `contains` is False for
        that pair repo-wide.
        """
        raise NotImplementedError

    def set_state(self, *, state: State) -> None:
        """External override: what happens when a human (via HumanOracle, called
        through Problem.execute_human_command) physically moves the real state --
        not the environment's own dynamics, and not a semantic reset."""
        self.current_state = state

    @abc.abstractmethod
    def hard_reset(self) -> None:
        """Reset to the initial state distribution. Only for the harness to call
        before a run starts -- never mid-practice, and never by the agent itself.
        Concrete implementations sample an initial state and call self.set_state on
        it."""
        raise NotImplementedError

    def reset_movables(self) -> bool:
        """External override, like `set_state` but *partial*: reposition whichever
        non-robot objects a human could tidy up, robot untouched. Called only via
        `HumanOracle.execute_movables_reset`/`Problem.execute_movables_reset`, from
        `HumanCubeBinResetRequested`.

        `False` by default (declined -- caller must not claim a reset happened);
        `Tossing3DEnvironment` overrides, returns True. Not a widening of
        `set_state`: that takes a full `State` for every object, this touches an
        unspecified domain-chosen subset and takes no argument at all."""
        return False

    def set_substep_recording(self, *, enabled: bool) -> None:
        """Turn on per-substep frame capture for domains whose dynamics run at a
        finer time granularity than one frame per `take_action` call -- a no-op by
        default, like `reset_movables`. `Tossing3DEnvironment` overrides: one
        `take_action` there is a whole controller execution over hundreds of MuJoCo
        ticks, so a caller recording practice video (`recording.period_recorder.
        PeriodRecorder`) needs the ticks in between, not just the boundary state.
        Every other domain here draws its `core.Renderer` frame from `State` alone,
        so there is nothing finer-grained to capture and this stays a no-op."""
        return None

    def drain_substep_frames(self) -> list[np.ndarray]:
        """Every frame collected since the last drain (or since
        `set_substep_recording` last turned collection on), clearing the buffer.

        Empty by default, matching `set_substep_recording`'s no-op default: a
        caller that always drains, whether or not this domain has anything finer
        than one frame per transition, gets an empty list rather than an
        AttributeError. `Tossing3DEnvironment` overrides to forward to
        `KinderBackend.drain_substep_frames`."""
        return []
