"""Eval-time choices that prefer the interior of a success region over its edge.

A prototype, not yet wired into `LearnedSkillSampler`. Both pieces learn only from
success/failure labels.

Why the plain argmax picks edges: `MlpBinaryClassifier` is a ReLU net with a sigmoid,
trained on BCE without regularization. On a handful of nearly separable labels the
loss is minimized by growing the logits without bound, so the predicted probability
saturates -- in float32, to exactly 1.0 -- over a broad region that reaches right up
to the nearest failure. `LearnedSkillSampler.sample` then breaks the tie uniformly
over every saturated candidate (its deviation 7), so a candidate one step inside the
boundary is exactly as likely to be chosen as one deep inside it. Nothing in the
score carries distance-to-boundary.

- `NoiseSmoothedScore` scores a candidate by the mean of any score function over an
  execution-noise kernel around it, so a candidate scores well only if its
  neighbourhood does. The kernel width is the margin and is an input, not learned.
- `LaplaceGpClassifier` is a Gaussian-process classifier whose latent slope is
  finite: repeated commands with mixed outcomes pull its probability below 1, so it
  peaks where the labels are most consistently successes. Its lower confidence bound
  additionally discounts regions with little data.
"""

from collections.abc import Callable

import numpy as np
from pydantic import BaseModel, PrivateAttr


class NoiseSmoothedScore(BaseModel):
    """Expected score under Gaussian execution noise added to each row column.

    `noise_std[j]` is the standard deviation added to column `j` of every row, in
    that column's own units; give 0 for the bias and any column that is not a
    commanded parameter. The same `num_draws` offsets are used for every row (common
    random numbers), so two candidates are ranked against identical noise.
    """

    noise_std: tuple[float, ...]
    num_draws: int = 256
    seed: int = 0

    def smoothed_scores(
        self, *, rows: np.ndarray, score_rows: Callable[[np.ndarray], np.ndarray]
    ) -> np.ndarray:
        rows = np.asarray(rows, dtype=np.float64)
        if rows.ndim != 2 or rows.shape[1] != len(self.noise_std):
            raise ValueError(
                f"noise_std has {len(self.noise_std)} entries but rows have shape {rows.shape}."
            )
        rng = np.random.default_rng(self.seed)
        offsets = rng.standard_normal((self.num_draws, rows.shape[1])) * np.asarray(self.noise_std)
        perturbed = (rows[None, :, :] + offsets[:, None, :]).reshape(-1, rows.shape[1])
        scores = np.asarray(score_rows(perturbed), dtype=np.float64)
        return scores.reshape(self.num_draws, rows.shape[0]).mean(axis=0)


class LaplaceGpClassifier(BaseModel):
    """Binary GP classifier, logistic likelihood, Laplace approximation.

    Rasmussen and Williams, Algorithms 3.1 and 3.2, with a squared-exponential kernel
    on inputs rescaled to [0, 1] by `lower`/`upper`. The constant prior mean is the
    logit of the Laplace-smoothed overall success rate, so unexplored space reverts
    to "as good as average" rather than to 1/2. Length scale and signal standard
    deviation are chosen by approximate marginal likelihood over a small grid.
    """

    lower: tuple[float, ...]
    upper: tuple[float, ...]
    length_scales: tuple[float, ...] = (0.08, 0.12, 0.18, 0.27, 0.4)
    signal_stds: tuple[float, ...] = (1.0, 2.0, 3.0)
    newton_iters: int = 50

    _x: np.ndarray | None = PrivateAttr(default=None)
    _prior_mean: float = PrivateAttr(default=0.0)
    _length_scale: float = PrivateAttr(default=0.2)
    _signal_std: float = PrivateAttr(default=1.0)
    _residual: np.ndarray | None = PrivateAttr(default=None)
    _sqrt_w: np.ndarray | None = PrivateAttr(default=None)
    _chol: np.ndarray | None = PrivateAttr(default=None)

    @property
    def length_scale(self) -> float:
        return self._length_scale

    @property
    def signal_std(self) -> float:
        return self._signal_std

    def fit(self, *, x_data: np.ndarray, y_data: np.ndarray) -> None:
        x = self._normalize(x_data=x_data)
        y = np.asarray(y_data, dtype=np.float64)
        rate = (y.sum() + 1.0) / (len(y) + 2.0)
        self._prior_mean = float(np.log(rate / (1.0 - rate)))
        self._x = x
        best: tuple[float, float, float] | None = None
        for length_scale in self.length_scales:
            for signal_std in self.signal_stds:
                evidence = self._laplace(
                    x=x, y=y, length_scale=length_scale, signal_std=signal_std
                )[0]
                if best is None or evidence > best[0]:
                    best = (evidence, length_scale, signal_std)
        assert best is not None
        self._length_scale, self._signal_std = best[1], best[2]
        _, residual, sqrt_w, chol = self._laplace(
            x=x, y=y, length_scale=self._length_scale, signal_std=self._signal_std
        )
        self._residual, self._sqrt_w, self._chol = residual, sqrt_w, chol

    def latent(self, *, x_data: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Posterior mean and variance of the latent logit at each row."""
        if self._x is None or self._residual is None or self._sqrt_w is None:
            raise RuntimeError("LaplaceGpClassifier used before fit.")
        assert self._chol is not None
        xs = self._normalize(x_data=x_data)
        k_star = self._kernel(
            a=self._x, b=xs, length_scale=self._length_scale, signal_std=self._signal_std
        )
        mean = self._prior_mean + k_star.T @ self._residual
        v = np.linalg.solve(self._chol, self._sqrt_w[:, None] * k_star)
        variance = np.maximum(self._signal_std**2 - np.sum(v * v, axis=0), 1e-12)
        return mean, variance

    def predict_proba(self, *, x_data: np.ndarray) -> np.ndarray:
        """Predictive success probability (probit approximation to the averaged
        logistic)."""
        mean, variance = self.latent(x_data=x_data)
        return self._sigmoid(z=mean / np.sqrt(1.0 + np.pi * variance / 8.0))

    def lower_confidence_bound(self, *, x_data: np.ndarray, kappa: float) -> np.ndarray:
        """The sigmoid of the latent's mean minus `kappa` standard deviations."""
        mean, variance = self.latent(x_data=x_data)
        return self._sigmoid(z=mean - kappa * np.sqrt(variance))

    def _laplace(
        self, *, x: np.ndarray, y: np.ndarray, length_scale: float, signal_std: float
    ) -> tuple[float, np.ndarray, np.ndarray, np.ndarray]:
        n = len(y)
        k = self._kernel(a=x, b=x, length_scale=length_scale, signal_std=signal_std)
        f = np.zeros(n)
        chol = np.eye(n)
        sqrt_w = np.zeros(n)
        for _ in range(self.newton_iters):
            pi = self._sigmoid(z=f + self._prior_mean)
            w = pi * (1.0 - pi)
            sqrt_w = np.sqrt(w)
            chol = np.linalg.cholesky(np.eye(n) + sqrt_w[:, None] * k * sqrt_w[None, :])
            b = w * f + (y - pi)
            a = b - sqrt_w * np.linalg.solve(chol.T, np.linalg.solve(chol, sqrt_w * (k @ b)))
            f_new = k @ a
            if np.max(np.abs(f_new - f)) < 1e-8:
                f = f_new
                break
            f = f_new
        pi = self._sigmoid(z=f + self._prior_mean)
        residual = y - pi
        a = np.linalg.solve(k + 1e-9 * np.eye(n), f)
        log_lik = np.sum(y * np.log(pi + 1e-300) + (1.0 - y) * np.log(1.0 - pi + 1e-300))
        evidence = -0.5 * float(a @ f) + float(log_lik) - float(np.sum(np.log(np.diag(chol))))
        sqrt_w = np.sqrt(pi * (1.0 - pi))
        chol = np.linalg.cholesky(np.eye(n) + sqrt_w[:, None] * k * sqrt_w[None, :])
        return evidence, residual, sqrt_w, chol

    def _normalize(self, *, x_data: np.ndarray) -> np.ndarray:
        lower = np.asarray(self.lower, dtype=np.float64)
        span = np.asarray(self.upper, dtype=np.float64) - lower
        return (np.asarray(x_data, dtype=np.float64) - lower) / span

    @staticmethod
    def _kernel(
        *, a: np.ndarray, b: np.ndarray, length_scale: float, signal_std: float
    ) -> np.ndarray:
        sq = np.sum((a[:, None, :] - b[None, :, :]) ** 2, axis=-1)
        return signal_std**2 * np.exp(-0.5 * sq / length_scale**2)

    @staticmethod
    def _sigmoid(*, z: np.ndarray) -> np.ndarray:
        return 0.5 * (1.0 + np.tanh(0.5 * z))
