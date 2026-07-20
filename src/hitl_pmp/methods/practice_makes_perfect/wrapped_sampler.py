from collections.abc import Callable

import numpy as np
import torch
from pydantic import BaseModel, ConfigDict, Field

from hitl_pmp.core.method.types import GroundSkill, Skill
from hitl_pmp.core.problem.environment.types import Object, State


class MlpBinaryClassifier(BaseModel):
    """A small MLP binary classifier trained with BCE loss + Adam, matching
    predicators' MLPBinaryClassifier defaults exactly
    (mlp_classifier_hid_sizes=[32, 32], lr=1e-3, early-stopped after
    n_iter_no_change=5000 iterations with no loss improvement). A pydantic
    BaseModel wrapping a torch.nn.Module -- deliberately mutable/stateful
    (trained weights), mirroring SkillCompetenceModel's own justification for
    departing from this project's frozen-data-type norm.

    Normalizes inputs to zero mean/unit variance before training/prediction,
    matching predicators' _NormalizingBinaryClassifier (unconditional in their
    class hierarchy, not an experiment-specific ablation) -- Light Switch's own
    inputs genuinely need this: WrappedSampler._construct_input concatenates
    raw x-position features (up to grid_size, e.g. 100) with continuous params
    in [-1, 1], and training on that mismatched scale unnormalized would
    distort gradients. Deliberately does NOT rebalance classes the way
    predicators' mlp_classifier_balance_data can -- confirmed that flag is
    explicitly set to False in the real grid_row/EES config
    (scripts/configs/active_sampler_learning.yaml), so skipping it here is
    the faithful choice, not a gap."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    input_dim: int
    hidden_sizes: tuple[int, ...] = (32, 32)
    learning_rate: float = 1e-3
    max_iters: int = 100_000
    n_iter_no_change: int = 5_000
    seed: int = 0
    network: torch.nn.Module | None = None
    input_mean: np.ndarray | None = None
    input_std: np.ndarray | None = None

    def fit(self, *, inputs: np.ndarray, labels: np.ndarray) -> None:
        torch.manual_seed(self.seed)
        self.input_mean = inputs.mean(axis=0)
        self.input_std = inputs.std(axis=0) + 1e-6  # avoid dividing by zero on a constant feature
        normalized_inputs = (inputs - self.input_mean) / self.input_std

        layers: list[torch.nn.Module] = []
        in_size = self.input_dim
        for hidden_size in self.hidden_sizes:
            layers.append(torch.nn.Linear(in_size, hidden_size))
            layers.append(torch.nn.ReLU())
            in_size = hidden_size
        layers.append(torch.nn.Linear(in_size, 1))
        layers.append(torch.nn.Sigmoid())
        network = torch.nn.Sequential(*layers)

        x = torch.as_tensor(normalized_inputs, dtype=torch.float32)
        y = torch.as_tensor(labels, dtype=torch.float32).reshape(-1, 1)
        optimizer = torch.optim.Adam(network.parameters(), lr=self.learning_rate)
        loss_fn = torch.nn.BCELoss()

        best_loss = float("inf")
        iters_since_improvement = 0
        for _ in range(self.max_iters):
            optimizer.zero_grad()
            loss = loss_fn(network(x), y)
            loss.backward()
            optimizer.step()
            loss_value = float(loss.item())
            if loss_value < best_loss:
                best_loss = loss_value
                iters_since_improvement = 0
            else:
                iters_since_improvement += 1
            if iters_since_improvement >= self.n_iter_no_change:
                break

        self.network = network

    def predict_proba(self, *, x: np.ndarray) -> float:
        if self.network is None or self.input_mean is None or self.input_std is None:
            raise RuntimeError("MlpBinaryClassifier.predict_proba called before fit().")
        with torch.no_grad():
            normalized_x = (x - self.input_mean) / self.input_std
            tensor = torch.as_tensor(normalized_x, dtype=torch.float32).unsqueeze(0)
            return float(self.network(tensor).item())


class SamplerTrainingExample(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    state: State
    ground_skill: GroundSkill
    params: np.ndarray
    success: bool


class WrappedSampler(BaseModel):
    """Reranks a skill's base (prior) continuous-parameter sampler using a
    trained classifier over (state, objects, params) -> predicted success
    probability -- ported from predicators'
    _ClassifierWrappedSamplerLearner. At execution (test) time, draws
    num_samples candidates from the base sampler and picks the
    highest-scoring one; at exploration time, does the same but with
    probability exploration_epsilon picks a uniformly random candidate
    instead -- the paper's pi+ = eps*pi0 + (1-eps)*pi, matching predicators'
    epsilon-greedy _wrap_sampler_exploration exactly. One instance per Skill
    (not per GroundSkill/object grounding): the paper notes weight sharing
    across groundings has "negligible impact" specifically in Light Switch,
    so this intentionally skips that complexity rather than porting it."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    skill: Skill
    num_samples: int = 100  # active_sampler_learning_num_samples
    exploration_epsilon: float = 0.5  # active_sampler_learning_exploration_epsilon
    training_data: list[SamplerTrainingExample] = Field(default_factory=list)
    classifier: MlpBinaryClassifier | None = None

    def record(
        self, *, state: State, ground_skill: GroundSkill, params: np.ndarray, success: bool
    ) -> None:
        self.training_data.append(
            SamplerTrainingExample(
                state=state, ground_skill=ground_skill, params=params, success=success
            )
        )

    def retrain(self) -> None:
        """Refits the classifier from every example recorded so far. A no-op
        until there's at least one success AND one failure -- BCE loss is
        undefined over a single class, and predicators' own oracle-sampler
        baseline effectively starts the same way (no reranking preference
        until there's something to discriminate)."""
        labels = [float(example.success) for example in self.training_data]
        if len(set(labels)) < 2:
            return
        inputs = np.stack([
            WrappedSampler._construct_input(
                state=example.state, objects=example.ground_skill.objects, params=example.params
            )
            for example in self.training_data
        ])
        classifier = MlpBinaryClassifier(input_dim=inputs.shape[1])
        classifier.fit(inputs=inputs, labels=np.array(labels))
        self.classifier = classifier

    def sample_for_execution(
        self,
        *,
        ground_skill: GroundSkill,
        state: State,
        base_sample_fn: Callable[..., np.ndarray],
        rng: np.random.Generator,
    ) -> np.ndarray:
        candidates = [
            base_sample_fn(ground_skill=ground_skill, rng=rng) for _ in range(self.num_samples)
        ]
        if self.classifier is None:
            return candidates[0]
        scores = [
            self.classifier.predict_proba(
                x=WrappedSampler._construct_input(
                    state=state, objects=ground_skill.objects, params=candidate
                )
            )
            for candidate in candidates
        ]
        return candidates[int(np.argmax(scores))]

    def sample_for_exploration(
        self,
        *,
        ground_skill: GroundSkill,
        state: State,
        base_sample_fn: Callable[..., np.ndarray],
        rng: np.random.Generator,
    ) -> np.ndarray:
        if rng.uniform() < self.exploration_epsilon:
            return base_sample_fn(ground_skill=ground_skill, rng=rng)
        return self.sample_for_execution(
            ground_skill=ground_skill, state=state, base_sample_fn=base_sample_fn, rng=rng
        )

    @staticmethod
    def _construct_input(
        *, state: State, objects: tuple[Object, ...], params: np.ndarray
    ) -> np.ndarray:
        # No feature engineering (matches the paper's own Light Switch setup,
        # confirmed against Appendix G's ablation description): the full raw
        # feature vector of every bound object, plus the candidate params.
        return np.concatenate([*(state[obj] for obj in objects), params])
