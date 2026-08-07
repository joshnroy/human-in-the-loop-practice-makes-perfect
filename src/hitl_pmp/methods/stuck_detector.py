from pydantic import BaseModel, Field, PrivateAttr

from hitl_pmp.core.problem.environment.types import State

# One observed state, reduced to something hashable: the objects' (name, type name)
# paired with the raw bytes of their feature vectors, sorted so dict insertion order
# cannot change it. A module-level alias rather than an inline annotation because it is
# written out three times below and reads as noise each time.
StateKey = tuple[tuple[str, str, bytes], ...]


class StuckDetector(BaseModel):
    """Whether practice has stopped reaching anywhere new: True once `patience`
    consecutive steps have all landed in states already visited in this stretch.

    **This is the project's operational definition of "stuck", and it is a choice with
    real alternatives.** See the module's own tests and
    `docs/experiment-logs/` for the arms it drives. The thing being detected is an
    *absorbing region*: the robot is still acting, still changing the state, and can no
    longer reach anything it has not already reached. Tossing Room's one-way ledge is the
    concrete case -- rooms 0-2 are severed from the item pile in room 3, so a practice
    period that steps left once can never pick anything up again.

    **It lives method-side, under `methods/`, because deciding when to ask for a human
    is part of the method and not part of the harness.** An earlier version of this
    class sat beside `practice_loop.py` and was described here as "a harness-side
    proxy"; that was the design Josh rejected -- a monitor watching the state stream and
    summoning a human on the robot's behalf measures the monitor. `methods/
    help_seeking.py` is the wrapper that turns this signal into a `HumanHelpRequested`
    the `Method` itself raises. What is still true is that this class needs nothing
    domain-specific and nothing from any particular `Method` -- but that is now a
    statement about *reuse*, i.e. any `Method` can compose it, rather than a
    justification for putting it in the loop.

    **Why novelty and not the two other signals available.**

      * **`state != previous_state` is too weak.** A robot stranded behind the ledge
        still walks: `MoveRoom` is applicable in every room, so every step changes the
        robot's `room` feature and a no-change test never fires. Measured on the
        reset-free A/B, the one-way reset-free arm spends its full 150 transitions per
        cycle walking. Novelty over the whole stretch catches the cycle that a
        one-step comparison cannot see.
      * **`InteractionComplete` is too strict.** `EesMethod` raises it only when *no*
        ground skill is applicable at all (`_practice_plan` falls back to a uniformly
        random applicable skill first). On Tossing Room `MoveRoom` is always applicable,
        so it essentially never fires there, and an arm triggered on it would be
        indistinguishable from an arm with no human at all.

    **What this deliberately excludes.** It is not a claim that the goal is unreachable
    -- that is undecidable from the harness, which holds no model of the domain. A
    period can be genuinely productive and still repeat states (a lucky policy looping a
    short solved cycle), and a period can be truly dead while still producing novel
    states (any domain with a free-running continuous feature -- a clock, a drifting
    sensor -- makes every state novel forever, and this detector then never fires). It is
    honest in one direction only: a novel state really does mean something new was
    reached.

    **Exact float equality**, via `ndarray.tobytes()`. Two states that differ in the last
    bit of one feature are two states. That is right for a domain whose features are
    integers stored as floats (every Tossing Room feature except the item weight) and
    increasingly weak as a domain becomes continuous -- a tolerance would need a
    per-feature scale the harness does not have, so the cruder rule is stated rather
    than guessed at.

    A real per-run instance, not a static-method container: the visited set and the
    counter are genuine state that has to survive between calls (CLAUDE.md's dividing
    line). Both are `PrivateAttr`s rather than fields -- they are working memory, not
    configuration, and nothing serializes a detector."""

    # How many consecutive already-visited states count as stuck. 1 means "the very first
    # repeat", which is far too eager for anything real; the method-CLI's own
    # --stuck-patience default is set from the domain's cycle length instead.
    patience: int = Field(ge=1)

    _seen: set[StateKey] = PrivateAttr(default_factory=set)
    _steps_since_novel: int = PrivateAttr(default=0)

    @property
    def steps_since_novel(self) -> int:
        """Consecutive observed states that had all been visited before. 0 immediately
        after a novel state, and 0 on a fresh detector -- so it is a count of *repeats*,
        not of steps."""
        return self._steps_since_novel

    def observe(self, *, state: State) -> None:
        """Record a state the practice period has been in.

        **Which state, exactly, is the caller's business, and it moved.** A `Method`
        sees the state only when it is asked for an action, so `help_seeking.py` calls
        this with the state *before* each action -- s0, s1, s2, ... -- where the old
        harness-side caller passed the state *after* it, s1, s2, .... The two sequences
        differ only by the period's own opening state being prepended, so the same
        absorbing region produces the same verdict at the same moment; the method-side
        prefix is one element longer, which can only make a later state non-novel and
        never novel, so it is never the *less* eager of the two. This class needs no
        change either way -- it is a rule about a sequence, not about the loop.

        Idempotent in the sense that matters: observing the same state twice is exactly
        the input that makes it stuck."""
        key = StuckDetector.state_key(state=state)
        if key in self._seen:
            self._steps_since_novel += 1
        else:
            self._seen.add(key)
            self._steps_since_novel = 0

    def is_stuck(self) -> bool:
        return self._steps_since_novel >= self.patience

    def restart(self) -> None:
        """Begin a fresh stretch: forget the visited set as well as the counter.

        Forgetting the set is the load-bearing half. The caller restarts this whenever
        something *external* has written the state -- a new practice period, or a human
        rescue the Method was told about through `Method.observe_help_granted` -- and
        after a rescue the robot is by construction back somewhere it has already been.
        Keeping the set would make that state instantly non-novel and re-fire the rule
        within `patience` steps, so a rescued robot would ask again immediately and
        forever."""
        self._seen.clear()
        self._steps_since_novel = 0

    @staticmethod
    def state_key(*, state: State) -> StateKey:
        """A hashable, order-independent reduction of a `State`.

        Keyed by `(object name, type name)` rather than by name alone because a domain
        may carry two objects with one name under different types -- `tossingroom` does
        exactly that, holding both `trash: trash_type` and `trash: item_type` so that
        `--unsplit-skills` can select a typing without renaming anything. Only one of the
        two is ever present in a live `State`, but keying on the name alone would make
        the two indistinguishable if that ever stopped being true.

        Public because a test asserts on it directly, and because rederiving it elsewhere
        would be a second definition of what "the same state" means."""
        return tuple(
            sorted(
                (obj.name, obj.type.name, features.tobytes())
                for obj, features in state.data.items()
            )
        )
