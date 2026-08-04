from pydantic import BaseModel, PrivateAttr


class TranslationCache(BaseModel):
    """Memoizes stage 1 of `FastDownwardPlanner.plan` -- PDDL text to SAS text -- for
    the lifetime of one run.

    **Why this is sound.** The three stages are ordered translate -> patch costs ->
    search, and the per-ground-skill costs EES varies are injected in stage *2*, by
    rewriting the already-translated SAS file. Stage 1 therefore reads nothing but
    the domain and problem PDDL and the FD `alias` (which selects translator options
    as well as search options, so it is part of the key even though nothing in this
    repo passes a non-default one yet), and Fast Downward's translator is a
    deterministic function of those (verified: 20 translations of the same pair under
    20 different `PYTHONHASHSEED` values produce one distinct SHA-256). Re-running it
    for a key already seen can only reproduce a byte-identical SAS file.

    **Why it is worth having.** The evaluation test set is drawn once and replayed by
    every sweep (`practice_loop.py`), and practice replans toward the same handful of
    candidate preconditions over and over, so the *symbolic* inputs repeat even
    though the costs never do. Measured on Tossing Room / EES: 304 plan calls drew
    on **12** distinct (domain, problem) pairs -- a 96% hit rate -- while the
    cost-vector-inclusive key repeated only 21% of the time, which is exactly why
    the cache sits at the translate stage and not around `plan()` as a whole.

    **Failures are cached too, and they are the common case**: 207 of those 304 calls
    were unreachable goals that the translator rejects before search ever starts (EES
    scores practice candidates by trying to plan to each one's preconditions in
    turn). A hit on such an entry must re-raise, so `sas_str is None` records "the
    translator aborted" and is stored alongside the FD output needed to reconstruct
    the same `PlanningFailure` message. Silently returning "no SAS" instead would
    turn an unreachable goal into an empty plan.

    Only *that* kind of failure is stored, though -- FD's own "Driver aborting"
    verdict on the PDDL, which is reproducible. A translation that produced no SAS
    file and no abort message did not finish (it hit the timeout and was killed), and
    that outcome is a property of the machine at that instant rather than of the
    input. Caching it would make one transient timeout a permanent answer for that
    symbolic state for the rest of the run, precisely under the saturated load this
    cache exists to serve. `FastDownwardPlanner._translate` therefore skips the store
    on that path, so a timeout costs exactly one plan call, as it did before this
    cache existed.

    An instance, not a class-level dict: a run's `Method` owns one (see
    `EesMethod._translation_cache`) and passes it in per call, so nothing is shared
    between runs, tests, or processes, and `FastDownwardPlanner` stays the
    stateless static-method container it is -- the same "take the one instance you
    need as an explicit argument" rule `HumanOracle`/`Renderer` follow.

    Unbounded by design. Its size is the number of distinct *symbolic* states a run
    visits, not the number of plan calls: 12 entries over a run's 300+ calls above,
    and each entry is a few kilobytes of text. A domain whose abstraction is large
    enough for that to matter would need an eviction policy, which is deliberately
    not guessed at here."""

    _results: dict[tuple[str, str, str], "TranslationResult"] = PrivateAttr(default_factory=dict)

    def get(self, *, alias: str, domain_str: str, problem_str: str) -> "TranslationResult | None":
        """The stored outcome for this exact (alias, domain, problem), or None if it
        is unseen. None is "never translated", distinct from a stored result whose
        `sas_str` is None ("translated, and the translator aborted")."""
        return self._results.get((alias, domain_str, problem_str))

    def put(
        self, *, alias: str, domain_str: str, problem_str: str, result: "TranslationResult"
    ) -> None:
        self._results[(alias, domain_str, problem_str)] = result

    def num_entries(self) -> int:
        """How many distinct keys are held -- the quantity that bounds this cache's
        memory, reported by the profiling in
        `docs/experiment-logs/2026-08-04-fd-planning-overhead.md`."""
        return len(self._results)


class TranslationResult(BaseModel):
    """What one run of Fast Downward's translator produced: the SAS text, or None if
    it aborted. `fd_output` is everything FD printed, kept so that a cache hit on an
    aborted translation raises the same `PlanningFailure` message the original miss
    did rather than a vaguer stand-in."""

    sas_str: str | None
    fd_output: str
