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
`mlp_classifier_n_iter_no_change = 5000`, `sampler_mlp_classifier_max_itr = 10000`)
as overridden by `scripts/configs/active_sampler_learning.yaml`
(`sampler_mlp_classifier_max_itr: 100000`, `mlp_classifier_balance_data: False`) --
the config the paper's own experiments were launched with. Note that
`sampler_mlp_classifier_max_itr` therefore has *two* reference values: predicators'
library default of 10000, and the paper launch config's 100000.

Scope: this file owns *only* parameter selection for a single skill. There is one
`LearnedSkillSampler` per skill *name* (parameterized option), never per grounding,
because `object_specific_samplers=False` is the paper's setting. Choosing *which*
skill to practice, and the competence models that consume `was_random`, live
elsewhere.

Deviations from predicators, all deliberate:

1. `max_train_iters` defaults to 1000 on the two classes in this file, matching
   neither predicators' settings.py default (10000) nor the paper config's 100000.
   100000 full-batch steps per skill per learning cycle is minutes of CPU per refit
   and makes the test suite unusable, so these classes keep the cheap value.
   Nothing else about the optimizer differs.

   That default is never reached in a real run: `EesMethod._refit_samplers` always
   passes `max_train_iters=self.sampler_max_train_iters` explicitly, and the only
   code that constructs `LearnedSkillSampler`/`MlpBinaryClassifier` without it is
   `tests/methods/practice_makes_perfect/`, which overrides it anyway. So the
   observation that 1000 sits below `n_iter_no_change` -- which is what motivated
   raising `EesMethod.sampler_max_train_iters` off 1000 -- is harmless here: unit
   tests want a fixed cheap step count, not early stopping.

   Do NOT read this as "raise it to 100000 for real experiments", which is what an
   earlier version of this docstring said. Real runs come through
   `EesMethod.sampler_max_train_iters`, whose default of 10000 is predicators' own
   settings.py default. More training measurably overfits the decisive Ball-Ring
   cup-placement classifier: train BCE 5.9e-3 at 10000 against 2.8e-5 at 100000
   (i.e. it interpolates the training set) while held-out argmax success falls from
   0.988 to 0.930 (paired, t = 5.67, 10/10 seeds).
   See docs/experiment-logs/2026-08-03-ballring-iters.md.
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
6. Whenever the classifier cannot *discriminate* among the candidates it was
   handed, `sample` returns a *uniformly random* candidate rather than the argmax
   of a degenerate score vector. This matches what predicators effectively does
   pre-learning -- the NSRT's own base sampler is used unwrapped, i.e. a single
   unfiltered draw -- and avoids silently biasing every early episode toward
   whichever candidate the caller happened to draw first.
   `was_random` still reports `False` there: it means specifically "the
   epsilon-greedy branch fired", which is the signal the competence models key on,
   and a sampler with nothing to say has no greedy branch to deviate from.
   `was_informed` is the flag that reports the fallback fired; see `SamplerChoice`
   for why the two are kept separate.

   The condition is a property of the *scores*, not of the training set. Gating on
   how much data exists instead would be wrong: measured on `ThrowRecycling`, a
   one-positive classifier's argmax still lands 97/275, above the 1-in-5 a uniform
   draw gets, because the harm is concentrated in the 170/275 of refits whose
   decision boundary came out backwards. A count-based gate would discard the good
   half with the bad.

   The cost of this branch, stated plainly: the fallback draws over *all*
   candidates, so a plateau that covers most but not all of them has its implied
   ranking thrown away, and a fitted classifier's uniform draw is now fed to the
   competence models as a deliberate attempt. That is intended -- a plateau this
   wide is the signature of the saturated box whose orientation one positive
   cannot pin down, and a wrong-slope classifier's argmax lands 38/170, the 1-in-5
   of a uniform draw -- but it is a real trade against `was_random`'s stated
   rationale, not a free win. Suppressing the competence update instead would pin
   a never-successful skill at its Beta(10, 1) prior forever, which is worse.
7. `sample` breaks a tie among equally-scored candidates *uniformly at random*
   rather than taking the lowest index. `np.argmax` returns the first maximal
   index, which on a saturated classifier -- and the ported architecture does
   saturate, interpolating <= 16 rows to a train BCE of ~5e-6 -- means the pick is
   decided by the caller's draw order while `was_random` reports a deliberate
   choice. Measured, 91/275 of one-positive probes were such ties.

   This branch alone is *distribution-preserving*: the candidates are iid and
   therefore exchangeable, so conditional on the multiset of candidates, "the first
   one attaining the maximum" and "a uniform draw among those attaining it" have
   the same law. It removes an order dependence and corrects a mislabelling without
   changing the distribution of the parameters returned. **Deviation 6's fallback
   is not** -- see above; it deliberately widens the draw from the plateau to the
   whole candidate set, and the support of the output therefore jumps
   discontinuously at `uninformative_tie_fraction`.

Neither deviation detects a classifier that is flat to within a rounding error but
not bit-identical: the guard is exact equality against `scores.max()`. All three
measured degenerate cases (unfitted, both single-class shortcuts, and the saturated
float32 sigmoid) do produce exactly equal scores, so it covers them -- but "cannot
discriminate" is the broader property, and this detects only its exact form.
"""

import contextlib
import copy
import math
from collections.abc import Iterator

import numpy as np
import torch
from pydantic import BaseModel, ConfigDict, PrivateAttr, model_validator
from torch import nn

from hitl_pmp.core.method.types import SamplerConsultation


class SingleThreadedTorch:
    """Runs a block of torch arithmetic at exactly one intra-op thread.

    `torch.manual_seed` pins the *initial weights* but not the **reduction order** of
    the matmuls that follow: torch splits a dot product across intra-op threads and
    sums the partial results in whatever order they finish, and float addition is not
    associative. So the same seed on the same data trained to different weights
    depending on how many threads the process happened to have -- making the ambient
    thread count a second, unrecorded input to every result.

    That is not hypothetical here. `scripts/run_sweep.py` pins `OMP_NUM_THREADS=1` on
    every child it spawns, while a bare `python -m hitl_pmp.cli` run inherits the
    machine's default (24 on this box), so a sweep and a CLI re-run of the *same seed*
    were two different experiments. See `docs/tossing3d-integration-status.md` section
    5.9, where exactly that comparison nearly became a false "concurrency perturbs
    Tossing3D" finding.

    Pinning here rather than at the CLI boundary keeps the guarantee a property of the
    sampler itself, so it holds however the run was launched -- including from a test,
    a notebook, or an `analysis/` script. The cost is negligible: these nets are
    32x32 and the wall clock on any real run is dominated by the simulator.
    """

    @staticmethod
    @contextlib.contextmanager
    def scope() -> Iterator[None]:
        previous = torch.get_num_threads()
        torch.set_num_threads(1)
        try:
            yield
        finally:
            torch.set_num_threads(previous)


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
    # See deviation 1 in the module docstring: predicators' settings.py default is
    # 10000 and the paper config uses 100000. Real runs always pass this explicitly.
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
        # Scoring is pinned for the same reason training is: this forward pass decides
        # which candidate the sampler returns, so a thread-dependent reduction here
        # would move the argmax even against identical weights.
        with SingleThreadedTorch.scope(), torch.no_grad():
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
        with SingleThreadedTorch.scope():
            self._train_single_threaded(tensor_x=tensor_x, tensor_y=tensor_y, x_data=x_data)

    def _train_single_threaded(
        self, *, tensor_x: torch.Tensor, tensor_y: torch.Tensor, x_data: np.ndarray
    ) -> None:
        """The training loop proper. Split out only so `_train` reads as
        seed-then-pin-then-train rather than nesting the whole loop in a `with`."""
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
        arithmetic, and it keeps this file free of a second stateful class.

        `hid_sizes=()` is a supported, deliberate configuration, not an edge case to
        "fix" by requiring at least one hidden layer: the loop below never executes,
        so the built net is exactly `[nn.Linear(input_dim, 1), nn.Sigmoid()]` --
        logistic regression, zero hidden layers, zero ReLUs. `EesMethod.sampler()`
        relies on exactly this to implement its `sampler_classifier="linear"`
        ablation (see that method's docstring): reusing this class rather than
        hand-writing a second one is what guarantees the linear arm is byte-identical
        to the MLP arm in everything except the net's shape -- normalization, Adam,
        learning rate, `n_iter_no_change` early stopping, best-loss checkpointing,
        the single-class shortcut, and deviations 6/7's tie-breaking in
        `LearnedSkillSampler.sample` all stay shared code. Do not add a lower bound on
        `len(hid_sizes)` here or in `LearnedSkillSampler.hid_sizes`."""
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
            sampler.observe(sampler_input=..., success=...)   # a prebuilt input row
        sampler.fit()                      # refit from scratch on *all* data
        candidates = [base_sampler() for _ in range(100)]
        choice = sampler.sample(
            sampler_inputs=[build_row(c) for c in candidates],
            candidates=candidates,
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
    # Deviation 6: the maximum score being attained by *more* than this fraction of
    # the candidates counts as "the classifier cannot discriminate here", and the
    # pick falls back to a uniform draw over all candidates. At the default, a
    # score vector whose argmax is a tie over more than half the candidate set is
    # treated as carrying no information -- which is what the ported architecture
    # produces in the low-positive regime (median tie plateau 23.4% of the axis at
    # one positive, but 91/275 probes above half) and essentially never produces
    # once it has real data (4/885 probes at 9-16 positives). A degenerate score
    # vector -- unfitted (all 0.5) or the single-class shortcut (all 0.0 / all 1.0)
    # -- ties over *every* candidate and so always lands here.
    uninformative_tie_fraction: float = 0.5
    hid_sizes: tuple[int, ...] = (32, 32)
    # See deviation 1: cheap test-only default, always overridden by EesMethod.
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

        This is only the *default* ("all") row layout: the caller (`EesMethod`) is
        what actually builds each classifier input row, so a domain that does oracle
        feature selection (`SkillProvider.oracle_sampler_input`) supplies a curated
        row instead. Either way the sampler below consumes an already-built row, so it
        stays domain-agnostic about which features a row contains.
        """
        return [1.0, *state_features, *(float(p) for p in params)]

    @property
    def is_fitted(self) -> bool:
        return self._classifier.is_fitted

    @property
    def num_observations(self) -> int:
        return len(self._labels)

    def observed_inputs(self) -> list[list[float]]:
        """A copy of every training row observed so far -- `num_observations`' content
        counterpart. Read-only by construction (a fresh list of fresh rows), so a caller
        cannot reach into the training set through it.

        Exists because "this sampler saw only its own skill's data" is a claim an
        experiment can rest on and a count alone cannot settle: two samplers can hold
        the same number of rows and still have been fed each other's. See
        `tests/environments/tossingroom/test_sampler_separation.py`."""
        return [list(row) for row in self._inputs]

    def observe(self, *, sampler_input: list[float], success: bool) -> None:
        """Record one (already-built classifier input row) -> success transition.

        `sampler_input` is the full row the caller built for the chosen parameters at
        the state the skill was executed in (bias term included) -- either the default
        `build_sampler_input` layout or a domain's oracle row. Taking the prebuilt row
        rather than `(features, params)` is what lets one code path serve both feature
        selections, and -- critically -- lets the caller snapshot the row at *decision*
        time so a training row can never desync from the row that was scored (the state
        has already mutated by the time an outcome is observed).

        `success` is the label predicators computes as "did the ground skill's add
        effects hold in the resulting state" (`_ClassifierWrappedSamplerLearner`
        consumes the pre-labeled `_OptionSamplerDataset`); deciding that is the
        caller's job, not this file's.
        """
        self._inputs.append(list(sampler_input))
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

    def score_inputs(self, *, sampler_inputs: list[list[float]]) -> list[float]:
        """Predicted success probability for each already-built classifier input row --
        predicators' `_classifier_to_score_fn` composed with
        `_vector_score_fn_to_score_fn`. Returns 0.5 for everything when unfitted, so
        callers that only want to inspect scores get a well-defined, unopinionated
        answer instead of an exception."""
        if not sampler_inputs:
            raise ValueError(f"{self.skill_name}: scoring requires at least one input row.")
        if not self.is_fitted:
            return [0.5] * len(sampler_inputs)
        x_data = np.array(sampler_inputs, dtype=np.float64)
        return [float(p) for p in self._classifier.predict_proba(x_data=x_data)]

    def sample(
        self,
        *,
        sampler_inputs: list[list[float]],
        candidates: list[np.ndarray],
        explore: bool,
    ) -> "SamplerChoice":
        """Choose one candidate parameter vector; return it alongside how it was chosen.

        `candidates[i]` is the raw parameter vector (what the caller will realize into
        an action) and `sampler_inputs[i]` is its already-built classifier input row --
        the two are kept separate on purpose: the classifier may score a *transformed*
        view of the parameters (e.g. an oracle row that uses converted placement
        coordinates), but `sample` must return the untransformed parameter vector the
        caller drew, or `compute_action` would receive the wrong representation.

        `explore=False` is predicators' `_wrap_sampler_test`: the argmax of the
        classifier scores, epsilon never consulted, `was_random` always `False`.
        `explore=True` is `_wrap_sampler_exploration` with
        `strategy="epsilon_greedy"`: the same argmax unless `rng.uniform() <=
        exploration_epsilon`, in which case a uniformly random candidate is returned
        with `was_random` set. The caller is expected to suppress its competence
        update when `was_random` is `True`.

        That argmax is not predicators' `np.argmax`, in two ways, and both matter --
        see deviations 6 and 7. Ties for the best score are broken uniformly rather
        than at the lowest index; and when the scores do not discriminate at all, a
        uniformly random candidate is returned with `was_random` **`False`** and
        `was_informed` `False`. `SamplerChoice` says why those are two separate flags
        rather than one.

        Note the degenerate configuration: with a single candidate there is nothing
        to discriminate between, so every draw takes deviation 6's branch and
        `was_random` can never be `True`. `num_candidates` is 100 in every real run.
        """
        if not candidates:
            raise ValueError(f"{self.skill_name}: sample requires at least one candidate.")
        if len(sampler_inputs) != len(candidates):
            raise ValueError(
                f"{self.skill_name}: got {len(sampler_inputs)} input rows for "
                f"{len(candidates)} candidates; they must correspond one-to-one."
            )
        for candidate in candidates:
            if candidate.shape != (self.param_dim,):
                raise ValueError(
                    f"{self.skill_name}: expected candidates of shape ({self.param_dim},), "
                    f"got {candidate.shape}."
                )
        scores = np.asarray(self.score_inputs(sampler_inputs=sampler_inputs), dtype=np.float64)
        # The candidates attaining the maximum. An unfitted sampler (all 0.5) and the
        # single-class shortcut (all 0.0 or all 1.0) both put every candidate in here.
        # A NaN anywhere makes the comparison all-False and empties `best`; that is a
        # score vector nothing can be ranked by, so it takes the fallback rather than
        # indexing into an empty array.
        best = np.flatnonzero(scores == scores.max())
        if len(best) == 0 or len(best) > self.uninformative_tie_fraction * len(candidates):
            # Deviation 6: no discrimination, so no greedy branch to deviate from.
            # `was_random` stays False so the competence models still count this
            # outcome -- a skill whose sampler has learned nothing is a skill whose
            # competence *should* fall, and suppressing the update here would pin a
            # never-successful skill at its prior forever.
            index = int(self._rng.integers(0, len(candidates)))
            return SamplerChoice(params=candidates[index], was_random=False, was_informed=False)
        # Deviation 7: break the remaining ties uniformly rather than at the lowest
        # index, which would be the caller's draw order.
        index = int(best[self._rng.integers(0, len(best))])
        was_random = False
        if explore and self._rng.uniform() <= self.exploration_epsilon:
            index = int(self._rng.integers(0, len(candidates)))
            was_random = True
        return SamplerChoice(
            params=candidates[index], was_random=was_random, was_informed=not was_random
        )


class SamplerChoice(BaseModel):
    """What `LearnedSkillSampler.sample` decided, and how.

    The two flags are orthogonal and are consumed by different things, which is why
    this is a model rather than a second `bool` on a tuple:

    - `was_random` means exactly "the epsilon-greedy branch fired". It is the signal
      the competence models key on (`EesMethod.observe_outcome` skips the update when
      it is set), and its meaning is unchanged from before deviations 6 and 7 existed.
    - `was_informed` means "the classifier's scores actually discriminated among the
      candidates, so this parameter vector reflects something it learned". It is an
      *analysis* signal: without it, a greedy draw made on a degenerate score vector
      is indistinguishable in the record from one a trained classifier chose, and any
      greedy-versus-random statistic silently pools the two. It is recorded through
      `EesMethod._SkillAttempt.was_informed_choice` into a skill-trace script's
      `informed_*` tallies (the Tossing Room one was retired with its domain). Pooling the
      two inverted a published conclusion once already: recycling's greedy draws
      landed 22/103 while the informed subset landed 11/56, which is its own
      epsilon-random rate.

    Exactly three of the four combinations are reachable. A draw can be neither
    (`was_random=False, was_informed=False`): that is deviation 6's fallback, an
    honest uniform draw that no competence model should discount but no analysis
    should count as evidence of learning. It can never be both.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    params: np.ndarray
    was_random: bool
    was_informed: bool

    @property
    def consultation(self) -> SamplerConsultation:
        """These two flags as the single pool `SkillPracticeTally` files the attempt
        into. The one place the mapping lives, so the tally and the flags can never
        drift apart.

        Only three of the four `SamplerConsultation` values are reachable from here:
        `NO_SAMPLER` describes a skill that never reaches `sample` at all, so its
        caller supplies it rather than this property."""
        if self.was_random:
            return SamplerConsultation.EPSILON_RANDOM
        if self.was_informed:
            return SamplerConsultation.INFORMED
        return SamplerConsultation.UNINFORMATIVE

    @model_validator(mode="after")
    def _an_epsilon_random_draw_is_never_informed(self) -> "SamplerChoice":
        if self.was_random and self.was_informed:
            raise ValueError(
                "was_random and was_informed cannot both be set: an epsilon-random "
                "draw ignores the classifier's scores by construction."
            )
        return self
