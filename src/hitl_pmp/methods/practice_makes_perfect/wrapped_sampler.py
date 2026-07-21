"""The learned ("wrapped") sampler from Practice Makes Perfect / EES.

Ported from the sibling `hitl-practice` fork of predicators:

- `predicators/approaches/active_sampler_learning_approach.py`
  - `_ClassifierWrappedSamplerLearner._learn_nsrt_sampler` (~L392-452): builds the
    (input, binary label) dataset and refits an MLP classifier per skill.
  - `_wrap_sampler_test` / `_wrap_sampler_exploration` (~L689-732): draw
    `active_sampler_learning_num_samples` candidate parameter vectors from the
    skill's base/oracle sampler, score every one with the classifier, take the
    argmax -- and, at exploration time only, with probability
    `active_sampler_learning_exploration_epsilon` take a uniformly random candidate
    instead and *report* that it did (the caller suppresses the competence update
    for epsilon-random choices, since a deliberately random action says nothing
    about the skill's competence).
- `predicators/utils.py::construct_active_sampler_input` (~L309-320): the input
  vector layout, `[1.0 bias] + concat(state[obj] for obj in objects) + params`
  under the default `active_sampler_learning_feature_selection="all"`.
- `predicators/ml_models.py`: `MLPBinaryClassifier` (~L1108),
  `PyTorchBinaryClassifier` (~L373), `_NormalizingBinaryClassifier` (~L301) and
  `_train_pytorch_model` (~L1251) -- architecture, normalization, single-class
  fallback, full-batch Adam + BCE training loop with best-loss checkpointing.

Defaults come from `predicators/settings.py` (`active_sampler_learning_num_samples
= 100`, `active_sampler_learning_exploration_epsilon = 0.5`,
`active_sampler_learning_object_specific_samplers = False`,
`mlp_classifier_hid_sizes = [32, 32]`, `learning_rate = 1e-3`, `weight_decay = 0`,
`mlp_classifier_n_iter_no_change = 5000`) as overridden by
`scripts/configs/active_sampler_learning.yaml` (`sampler_mlp_classifier_max_itr:
100000`, `mlp_classifier_balance_data: False`) -- the config the paper's own
experiments were launched with.

Scope: this file owns *only* parameter selection for a single skill. There is one
`LearnedSkillSampler` per skill *name* (parameterized option), never per grounding,
because `object_specific_samplers=False` is the paper's setting. Choosing *which*
skill to practice, and the competence models that consume `was_random`, live
elsewhere.

Deviations from predicators, all deliberate:

1. `max_train_iters` defaults to 1000, not the paper config's 100000. 100000
   full-batch steps per skill per learning cycle is minutes of CPU per refit and
   makes the test suite unusable; the caller should raise it for real experiments
   (`LearnedSkillSampler(..., max_train_iters=100000)`). Nothing else about the
   optimizer differs.
2. The best-loss checkpoint is kept in memory (`copy.deepcopy` of the state dict)
   rather than round-tripped through a `tempfile.NamedTemporaryFile` as
   `_train_pytorch_model` does. Same weights, no stray temp files.
3. `_fit` in predicators raises `RuntimeError` if no reinitialization try reaches
   `best_loss < 1`. Here that case keeps the best weights found and returns. With
   `n_reinitialize_tries = 1` (the default) predicators would simply crash a long
   unattended practice run on a numerical fluke, and BCE loss below 1 is
   essentially always reached anyway.
4. The classifier is not pickled to disk. predicators dumps
   `<save_id>.sampler_classifier` for offline analysis; run artifacts are
   `--output-dir`'s concern in this codebase, not the sampler's.
5. Candidate parameter vectors are passed *in* by the caller rather than drawn here
   from a base sampler. The base/oracle sampler belongs to the environment's
   `skills.py`, and this file must stay domain-agnostic; `num_candidates` is kept
   as the documented count the caller should draw (predicators'
   `active_sampler_learning_num_samples`).
6. Before any successful `fit` (no data at all, or all-one-class data that
   predicators refuses to train on), `sample` returns a *uniformly random*
   candidate rather than the argmax of a degenerate score vector. This matches what
   predicators effectively does pre-learning -- the NSRT's own base sampler is used
   unwrapped, i.e. a single unfiltered draw -- and avoids silently biasing every
   early episode toward whichever candidate the caller happened to draw first.
   `was_random` still reports `False` there: it means specifically "the
   epsilon-greedy branch fired", which is the signal the competence models key on,
   and an unfitted sampler has no greedy branch to deviate from.
"""

import copy
import math

import numpy as np
import torch
from pydantic import BaseModel, ConfigDict, PrivateAttr
from torch import nn


class MlpBinaryClassifier(BaseModel):
    """Port of predicators' `MLPBinaryClassifier` stack (`ml_models.py` L301-L497,
    L1108-L1154): min/max input normalization, a single-class shortcut, and a fully
    connected ReLU net trained full-batch with Adam on binary cross-entropy.

    Not an abstract interface with per-domain subclasses -- like `core/metrics/
    metrics.py`, every method here is already the one behavior this project needs,
    so it is used directly.

    The net is built lazily in `fit`, once the input dimensionality is known
    (predicators does the same in `_initialize_net`), so the classifier can be
    constructed before any data exists.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    seed: int = 0
    # predicators settings.py: mlp_classifier_hid_sizes = [32, 32].
    hid_sizes: tuple[int, ...] = (32, 32)
    # See deviation 1 in the module docstring: the paper config uses 100000.
    max_train_iters: int = 1000
    # predicators settings.py: learning_rate = 1e-3, weight_decay = 0.
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    # predicators settings.py: mlp_classifier_n_iter_no_change = 5000.
    n_iter_no_change: int = 5000
    # active_sampler_learning.yaml sets mlp_classifier_balance_data: False for the
    # sampler classifier, so downsampling the majority class is off by default.
    balance_data: bool = False
    # predicators settings.py: sampler_mlp_classifier_n_reinitialize_tries = 1.
    n_reinitialize_tries: int = 1

    _net: nn.Module | None = PrivateAttr(default=None)
    _input_shift: np.ndarray | None = PrivateAttr(default=None)
    _input_scale: np.ndarray | None = PrivateAttr(default=None)
    _single_class_prediction: float | None = PrivateAttr(default=None)

    @property
    def is_fitted(self) -> bool:
        """True once `fit` has produced *something* that can score inputs -- either a
        trained net or the single-class shortcut."""
        return self._net is not None or self._single_class_prediction is not None

    def fit(self, *, x_data: np.ndarray, y_data: np.ndarray) -> None:
        """Refit from scratch on all data. Mirrors
        `_NormalizingBinaryClassifier.fit`: single-class shortcut first, then
        optional balancing, then normalization, then `_fit`."""
        self._net = None
        self._input_shift = None
        self._input_scale = None
        self._single_class_prediction = None
        if x_data.shape[0] == 0:
            return
        # "If there is only one class in the data, then there's no point in
        # learning, since any predictions other than that one class could only be
        # generalization issues." (ml_models.py L329-339)
        if np.all(y_data == 0):
            self._single_class_prediction = 0.0
            return
        if np.all(y_data == 1):
            self._single_class_prediction = 1.0
            return
        if self.balance_data and len(y_data) // 2 > int(y_data.sum()):
            x_data, y_data = self._balance(x_data=x_data, y_data=y_data)
        # ml_models.py::_normalize_data -- shift by the per-feature min, scale by the
        # per-feature range clipped below at 1 (so a constant feature, e.g. the bias
        # term, is left alone instead of dividing by zero).
        shift = np.min(x_data, axis=0)
        scale = np.clip(np.max(x_data - shift, axis=0), 1.0, None)
        self._input_shift = shift
        self._input_scale = scale
        self._train(x_data=(x_data - shift) / scale, y_data=y_data)

    def predict_proba(self, *, x_data: np.ndarray) -> np.ndarray:
        """Probability of class 1 for each row of `x_data` (which is NOT normalized
        -- normalization happens here, as in `PyTorchBinaryClassifier.predict_proba`).

        Batched, unlike predicators' one-row-at-a-time `predict_proba`: the caller
        scores 100 candidates per decision and a single forward pass is the same
        arithmetic in a fraction of the wall time.
        """
        if self._single_class_prediction is not None:
            return np.full(x_data.shape[0], self._single_class_prediction, dtype=np.float64)
        if self._net is None or self._input_shift is None or self._input_scale is None:
            raise RuntimeError("MlpBinaryClassifier.predict_proba called before fit.")
        normalized = (x_data - self._input_shift) / self._input_scale
        tensor_x = torch.from_numpy(np.asarray(normalized, dtype=np.float32))
        with torch.no_grad():
            probabilities = self._net(tensor_x).squeeze(dim=-1)
        return probabilities.detach().cpu().numpy().astype(np.float64)

    def _balance(self, *, x_data: np.ndarray, y_data: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Port of `ml_models.py::_balance_binary_classification_data`: keep every
        positive and an equal-sized random subset of the negatives."""
        rng = np.random.default_rng(self.seed)
        positive_indices = np.flatnonzero(y_data == 1)
        negative_indices = np.flatnonzero(y_data == 0)
        kept_negatives = rng.choice(negative_indices, replace=False, size=len(positive_indices))
        keep = np.concatenate([positive_indices, kept_negatives])
        return x_data[keep], y_data[keep]

    def _train(self, *, x_data: np.ndarray, y_data: np.ndarray) -> None:
        """Port of `PyTorchBinaryClassifier._fit` + `_train_pytorch_model`: full-batch
        Adam on BCE, no minibatching (predicators notes this explicitly), keeping the
        lowest-loss weights seen and stopping early after `n_iter_no_change`
        non-improving iterations."""
        torch.manual_seed(self.seed)
        tensor_x = torch.from_numpy(np.asarray(x_data, dtype=np.float32))
        tensor_y = torch.from_numpy(np.asarray(y_data, dtype=np.float32))
        loss_fn = nn.BCELoss()
        best_overall_loss = math.inf
        best_overall_state: dict[str, torch.Tensor] | None = None
        for try_index in range(self.n_reinitialize_tries):
            # Reinitialization tries must not all draw the same weights, so the seed
            # is offset per try (predicators re-applies `_reset_weights`, which draws
            # fresh values from the already-advanced global torch RNG).
            torch.manual_seed(self.seed + try_index)
            net = self._build_net(input_dim=x_data.shape[1])
            optimizer = torch.optim.Adam(
                net.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay
            )
            net.train()
            best_loss = math.inf
            best_iteration = 0
            best_state: dict[str, torch.Tensor] = copy.deepcopy(net.state_dict())
            for iteration in range(self.max_train_iters + 1):
                predictions = net(tensor_x).squeeze(dim=-1)
                loss = loss_fn(predictions, tensor_y)
                loss_value = loss.item()
                if loss_value < best_loss:
                    best_loss = loss_value
                    best_iteration = iteration
                    best_state = copy.deepcopy(net.state_dict())
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                if iteration - best_iteration > self.n_iter_no_change:
                    break
            if best_loss < best_overall_loss:
                best_overall_loss = best_loss
                best_overall_state = best_state
                net.load_state_dict(best_state)
                net.eval()
                self._net = net
            # predicators' convergence check: BCE loss below 1 counts as success.
            # See deviation 3 -- failing every try keeps the best weights instead of
            # raising.
            if best_overall_loss < 1:
                break
        assert best_overall_state is not None

    def _build_net(self, *, input_dim: int) -> nn.Module:
        """Port of `MLPBinaryClassifier._initialize_net`/`forward`: ReLU between
        hidden layers, a single output unit, sigmoid on top. Expressed as an
        `nn.Sequential` rather than a hand-written `nn.Module` subclass -- identical
        arithmetic, and it keeps this file free of a second stateful class."""
        layers: list[nn.Module] = []
        previous_dim = input_dim
        for hidden_dim in self.hid_sizes:
            layers.append(nn.Linear(previous_dim, hidden_dim))
            layers.append(nn.ReLU())
            previous_dim = hidden_dim
        layers.append(nn.Linear(previous_dim, 1))
        layers.append(nn.Sigmoid())
        return nn.Sequential(*layers)


class LearnedSkillSampler(BaseModel):
    """The wrapped sampler for one skill: score candidate parameter vectors with a
    learned success classifier and return the best one.

    One instance per skill *name*, not per grounding -- predicators'
    `active_sampler_learning_object_specific_samplers = False`. The classifier
    therefore generalizes across every grounding of the skill, and the objects a
    particular grounding binds enter only through `features` (the concatenated
    per-object feature vectors, in the ground skill's object order).

    Typical use per learning cycle:

        for rollout in cycle_data:
            sampler.observe(features=..., params=..., success=...)
        sampler.fit()                      # refit from scratch on *all* data
        params, was_random = sampler.sample(
            features=..., candidates=[base_sampler() for _ in range(100)],
            explore=True,
        )

    `seed` follows the repo's seed-field + `PrivateAttr` + `model_post_init`
    convention (`environments/lightswitch/tasks.py::LightSwitchTasks`): the epsilon
    RNG is derived from it and never assigned directly.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    skill_name: str
    param_dim: int
    # predicators settings.py: active_sampler_learning_num_samples = 100. Advisory
    # here -- the caller draws the candidates (deviation 5) -- but kept so the count
    # lives with the rest of the sampler's configuration.
    num_candidates: int = 100
    # predicators settings.py: active_sampler_learning_exploration_epsilon = 0.5,
    # with active_sampler_learning_exploration_sample_strategy = "epsilon_greedy".
    exploration_epsilon: float = 0.5
    seed: int = 0
    hid_sizes: tuple[int, ...] = (32, 32)
    max_train_iters: int = 1000
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    balance_data: bool = False

    # default_factory, not a bare [] -- pydantic would deep-copy a mutable default
    # per instance in current versions, but relying on that is exactly the trap this
    # codebase's "constructor-injected instance state" rule is meant to avoid, and
    # two samplers sharing a training set would be a silent correctness bug.
    _inputs: list[list[float]] = PrivateAttr(default_factory=list)
    _labels: list[int] = PrivateAttr(default_factory=list)
    _classifier: MlpBinaryClassifier = PrivateAttr()
    _rng: np.random.Generator = PrivateAttr()

    def model_post_init(self, __context: object) -> None:
        self._rng = np.random.default_rng(self.seed)
        self._classifier = MlpBinaryClassifier(
            seed=self.seed,
            hid_sizes=self.hid_sizes,
            max_train_iters=self.max_train_iters,
            learning_rate=self.learning_rate,
            weight_decay=self.weight_decay,
            balance_data=self.balance_data,
        )

    @staticmethod
    def build_sampler_input(*, state_features: list[float], params: np.ndarray) -> list[float]:
        """`[1.0 bias] + state_features + params` -- predicators'
        `utils.construct_active_sampler_input` under the default
        `active_sampler_learning_feature_selection = "all"`.

        `state_features` is the caller's concatenation of `state[obj]` over the
        ground skill's objects, in the ground skill's own object order (that order is
        what makes a single per-skill-name classifier meaningful across groundings).
        A `@staticmethod` on the class rather than a module-level function, per this
        repo's static-method-container rule.
        """
        return [1.0, *state_features, *(float(p) for p in params)]

    @property
    def is_fitted(self) -> bool:
        return self._classifier.is_fitted

    @property
    def num_observations(self) -> int:
        return len(self._labels)

    def observe(self, *, features: list[float], params: np.ndarray, success: bool) -> None:
        """Record one (state, chosen params) -> success transition.

        `success` is the label predicators computes as "did the ground skill's add
        effects hold in the resulting state" (`_ClassifierWrappedSamplerLearner`
        consumes the pre-labeled `_OptionSamplerDataset`); deciding that is the
        caller's job, not this file's.
        """
        if params.shape != (self.param_dim,):
            raise ValueError(
                f"{self.skill_name}: expected params of shape ({self.param_dim},), "
                f"got {params.shape}."
            )
        self._inputs.append(self.build_sampler_input(state_features=features, params=params))
        self._labels.append(int(success))

    def fit(self) -> None:
        """Refit the classifier from scratch on every observation ever made.

        Refitting from scratch (rather than warm-starting) is what
        `_ClassifierWrappedSamplerLearner._learn_nsrt_sampler` does each cycle: it
        rebuilds `X_classifier`/`y_classifier` from the full dataset and constructs a
        brand-new `MLPBinaryClassifier`. Calling this with no data is a no-op, so a
        harness can fit unconditionally every cycle.
        """
        if not self._labels:
            return
        self._classifier = MlpBinaryClassifier(
            seed=self.seed,
            hid_sizes=self.hid_sizes,
            max_train_iters=self.max_train_iters,
            learning_rate=self.learning_rate,
            weight_decay=self.weight_decay,
            balance_data=self.balance_data,
        )
        self._classifier.fit(
            x_data=np.array(self._inputs, dtype=np.float64),
            y_data=np.array(self._labels, dtype=np.float64),
        )

    def score_candidates(
        self, *, features: list[float], candidates: list[np.ndarray]
    ) -> list[float]:
        """Predicted success probability for each candidate parameter vector --
        predicators' `_classifier_to_score_fn` composed with
        `_vector_score_fn_to_score_fn`. Returns 0.5 for everything when unfitted, so
        callers that only want to inspect scores get a well-defined, unopinionated
        answer instead of an exception."""
        if not candidates:
            raise ValueError(f"{self.skill_name}: sample requires at least one candidate.")
        if not self.is_fitted:
            return [0.5] * len(candidates)
        x_data = np.array(
            [
                self.build_sampler_input(state_features=features, params=params)
                for params in candidates
            ],
            dtype=np.float64,
        )
        return [float(p) for p in self._classifier.predict_proba(x_data=x_data)]

    def sample(
        self, *, features: list[float], candidates: list[np.ndarray], explore: bool
    ) -> tuple[np.ndarray, bool]:
        """Choose one candidate parameter vector; return it with a flag saying whether
        it was chosen by the epsilon-random branch.

        `explore=False` is predicators' `_wrap_sampler_test`: pure argmax of the
        classifier scores, epsilon never consulted, flag always `False`.
        `explore=True` is `_wrap_sampler_exploration` with
        `strategy="epsilon_greedy"`: argmax unless `rng.uniform() <=
        exploration_epsilon`, in which case a uniformly random candidate is returned
        with the flag set. The caller is expected to suppress its competence update
        when the flag is `True`.

        When unfitted, a uniformly random candidate is returned with the flag `False`
        -- see deviation 6 in the module docstring.
        """
        if not candidates:
            raise ValueError(f"{self.skill_name}: sample requires at least one candidate.")
        if not self.is_fitted:
            return candidates[int(self._rng.integers(0, len(candidates)))], False
        scores = self.score_candidates(features=features, candidates=candidates)
        index = int(np.argmax(scores))
        was_random = False
        if explore and self._rng.uniform() <= self.exploration_epsilon:
            index = int(self._rng.integers(0, len(scores)))
            was_random = True
        return candidates[index], was_random
