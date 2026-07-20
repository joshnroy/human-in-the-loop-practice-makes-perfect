from pydantic import BaseModel, Field


class SkillCompetenceModel(BaseModel):
    """Tracks and predicts one skill's competence across online-learning cycles,
    using the sliding-window Beta-Bernoulli model predicators calls
    "optimistic" -- the actual default used for grid_row/EES
    (skill_competence_model="optimistic" in
    hitl-practice/scripts/configs/active_sampler_learning.yaml, not the more
    principled EM/latent-variable alternative the paper also describes trying).
    Ported directly from
    hitl-practice/predicators/competence_models.py's
    OptimisticSkillCompetenceModel + utils.beta_bernoulli_posterior_mean.

    Deliberately mutable (cycle_observations grows on every observe()/
    advance_cycle() call) -- unlike this project's usual frozen data types,
    this mirrors State's own shape (a pydantic BaseModel carrying real behavior
    methods that mutate its own fields), not the separate static-method-
    container rule for stateless business logic (there's one of these per
    ground skill/operator, so it genuinely needs its own identity/state)."""

    cycle_observations: list[list[bool]] = Field(default_factory=lambda: [[]])  # type: ignore[arg-type]
    alpha: float = 10.0  # predicators' skill_competence_default_alpha_beta
    beta: float = 1.0
    window_size: int = 5  # skill_competence_model_optimistic_window_size
    recency_size: int = 5  # skill_competence_model_optimistic_recency_size
    initial_prediction_bonus: float = 0.5  # skill_competence_initial_prediction_bonus

    def observe(self, *, outcome: bool) -> None:
        """Record a success/failure from actually executing the skill."""
        self.cycle_observations[-1].append(outcome)

    def advance_cycle(self) -> None:
        """Called once per online-learning cycle, after re-learning."""
        self.cycle_observations.append([])

    def get_current_competence(self) -> float:
        """Estimate: Beta-Bernoulli posterior mean over the last window_size
        non-empty cycles (older/empty cycles don't count)."""
        nonempty_cycles = [cycle for cycle in self.cycle_observations if cycle]
        if not nonempty_cycles:
            return self.alpha / (self.alpha + self.beta)
        window = min(len(nonempty_cycles), self.window_size)
        recent_outcomes = [outcome for cycle in nonempty_cycles[-window:] for outcome in cycle]
        return self._posterior_mean(outcomes=recent_outcomes)

    def predict_competence(self, *, num_additional_data: int) -> float:
        """Extrapolate: before two cycles of data exist, optimistically bump
        the current estimate by a fixed bonus. After that, find the largest
        per-cycle success-rate swing among the last recency_size non-empty
        cycles, and optimistically assume num_additional_data more
        observations would repeat that swing."""
        nonempty_cycles = [cycle for cycle in self.cycle_observations if cycle]
        current_competence = self.get_current_competence()
        if len(nonempty_cycles) < 2:
            return min(1.0, current_competence + self.initial_prediction_bonus)
        start = max(0, len(nonempty_cycles) - self.recency_size)
        per_cycle_rates = [sum(cycle) / len(cycle) for cycle in nonempty_cycles[start:]]
        best_change = max(per_cycle_rates) - min(per_cycle_rates)
        gain = best_change * num_additional_data
        return min(1.0, max(1e-6, current_competence + gain))

    def _posterior_mean(self, *, outcomes: list[bool]) -> float:
        """https://gregorygundersen.com/blog/2020/08/19/bernoulli-beta/ -- the
        standard Beta-Bernoulli conjugate update, matching predicators'
        _beta_bernoulli_posterior_alpha_beta exactly."""
        num_observations = len(outcomes)
        num_successes = sum(outcomes)
        alpha_n = self.alpha + num_successes
        beta_n = self.beta + (num_observations - num_successes)
        return alpha_n / (alpha_n + beta_n)
