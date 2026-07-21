from pydantic import BaseModel, Field


class OptimisticSkillCompetenceModel(BaseModel):
    """Tracks and predicts a *single* skill's competence (probability of success)
    from its history of outcomes, split into re-learning cycles. Ported from
    predicators' OptimisticSkillCompetenceModel
    (../hitl-practice/predicators/competence_models.py), which is the model EES
    actually runs with (predicators' settings.py: skill_competence_model =
    "optimistic"); the paper's other models there (legacy, latent-variable) are
    not ported since EES never selects them.

    Data layout: cycle_observations is a list of lists of bools -- one inner list
    per re-learning cycle. observe() appends to the current (last) cycle;
    advance_cycle() opens a new empty one. The split matters because the whole
    point of the model is to measure *improvement between cycles*, which a single
    pooled success history cannot express. Cycles that stayed empty (the skill was
    never attempted that cycle) are filtered out everywhere before any windowing,
    so a run of idle cycles never pushes real data out of a window.

    get_current_competence() pools the outcomes of the last window_size nonempty
    cycles and returns the Beta-Bernoulli posterior mean

        (alpha + s) / (alpha + beta + n)

    for s successes out of n pooled outcomes (see predicators'
    utils.beta_bernoulli_posterior_mean). The default prior (alpha, beta) =
    (10, 1) is deliberately lopsided: with no data the model reports 10/11 ~=
    0.91, i.e. a brand-new skill is *assumed* nearly competent. That is what makes
    EES try a skill at all before it has evidence, and it is why the sliding
    window exists -- the prior has to be washed out by recent data rather than by
    a skill's entire lifetime history.

    predict_competence(num_additional_data=k) answers "what would competence be if
    we collected k more outcomes this cycle?". This is where "optimistic" comes
    from: it takes the *raw* per-cycle means (plain averages, deliberately with no
    prior -- see the deviation note below) of the last recency_size nonempty
    cycles, measures the largest spread between any two of them
    (max(means) - min(means)), and assumes that best-ever observed per-cycle
    improvement will simply repeat, k times over:

        predicted = clip(current + (max(means) - min(means)) * k, 1e-6, 1.0)

    Two consequences EES depends on. A skill that has been improving gets an
    inflated prediction, so practicing it looks worth the cost. A skill whose
    per-cycle means have plateaued (identical means -> best_change == 0) predicts
    exactly its current competence, so practicing it promises zero gain and EES
    stops -- that is the whole "stop practicing a maxed-out skill" behavior. With
    fewer than 2 nonempty cycles there is no between-cycle change to measure at
    all, so the model falls back to a flat, even more optimistic
    current + initial_prediction_bonus (capped at 1.0), which is what gets a
    never-practiced skill tried in the first place.

    Faithfulness notes (deviations from a naive reading of the source):
    - The per-cycle competences used for max/min are plain np.mean over each
      cycle's outcomes, with *no* Beta prior applied -- unlike
      get_current_competence. That asymmetry is in the reference implementation
      and is preserved here (applying the prior would shrink all per-cycle means
      toward 10/11 and crush the measured spread).
    - predicators computes those means with an `inference_window` local that is
      hardcoded to 1 (it is not CFG.skill_competence_model_lookahead), so the
      loop reduces exactly to "one mean per cycle over the last recency_size
      nonempty cycles". Inlined as such here rather than carrying a constant.
    - CFG.skill_competence_model_lookahead = 1 is the *caller's* argument: EES
      calls predict_competence(1). It is not a field of this model.
    - predicators returns np.clip(...)'s numpy float from predict_competence;
      this port casts to a plain Python float so the return type matches the
      annotation and the other method.

    Unlike the older static-ClassVar interfaces in core/, this is a real
    constructor-injected instance: one model per skill, each with its own
    history, so an EES explorer holds a dict[skill_name,
    OptimisticSkillCompetenceModel] with no shared mutable state."""

    window_size: int = 5
    recency_size: int = 5
    alpha: float = 10.0
    beta: float = 1.0
    initial_prediction_bonus: float = 0.5

    # One inner list per re-learning cycle; the last one is the current cycle.
    # default_factory (not a bare default) is what gives every instance its own
    # list-of-lists rather than one shared class-level object.
    cycle_observations: list[list[bool]] = Field(default_factory=lambda: [list[bool]()])

    @property
    def num_observations(self) -> int:
        """Total attempts of this skill across *all* cycles, including ones that
        have aged out of the sliding window. EES's explorer needs this raw count
        for its UCB-style exploration bonus (which is about how much has been
        tried, not how recently), so it deliberately ignores window_size."""
        return sum(len(cycle) for cycle in self.cycle_observations)

    def observe(self, *, success: bool) -> None:
        """Records one success/failure from running the skill, into the current cycle."""
        self.cycle_observations[-1].append(success)

    def advance_cycle(self) -> None:
        """Called after re-learning: opens a new (empty) cycle."""
        self.cycle_observations.append([])

    def get_current_competence(self) -> float:
        """Beta-Bernoulli posterior mean over the last window_size nonempty cycles."""
        recent = self._nonempty_cycles()[-self.window_size :]
        outcomes = [outcome for cycle in recent for outcome in cycle]
        return self._posterior_mean(outcomes=outcomes)

    def predict_competence(self, *, num_additional_data: int) -> float:
        """Competence if num_additional_data more outcomes were collected this cycle."""
        nonempty = self._nonempty_cycles()
        current = self.get_current_competence()
        if len(nonempty) < 2:
            return min(1.0, current + self.initial_prediction_bonus)
        means = [sum(cycle) / len(cycle) for cycle in nonempty[-self.recency_size :]]
        best_change = max(means) - min(means)
        return float(min(1.0, max(1e-6, current + best_change * num_additional_data)))

    def _nonempty_cycles(self) -> list[list[bool]]:
        return [cycle for cycle in self.cycle_observations if cycle]

    def _posterior_mean(self, *, outcomes: list[bool]) -> float:
        num_total = len(outcomes)
        num_successes = sum(outcomes)
        alpha_n = self.alpha + num_successes
        beta_n = (num_total - num_successes) + self.beta
        return float(alpha_n / (alpha_n + beta_n))
