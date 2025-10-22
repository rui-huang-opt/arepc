from numpy import float64
from numpy import array, mean, average, stack, unique, where
from numpy import ones, zeros, exp, sqrt, log, maximum, minimum
from numpy.linalg import norm
from numpy.typing import NDArray
from numpy.random import choice


def min_f(probs: NDArray[float64], f: int) -> float | None:
    """
    The min-f score is defined as the f-th smallest unique score in the array,
    but it is not allowed to be the largest score.
    If there is only one unique score, return None to indicate that all scores are equal.
    """

    if probs.size == 0:
        raise ValueError("Probabilities array is empty.")

    deduped_probs = unique(probs)

    if len(deduped_probs) == 1:
        return None

    if f >= len(deduped_probs):
        return deduped_probs[-2]
    else:
        return deduped_probs[f - 1]


def loss_func(j: str, x_js: dict[str, NDArray[float64]]) -> float:
    return mean([norm(x_js[j] - x_js[k]) for k in x_js], dtype=float64)


class MWUpdate:
    def __init__(self, experts: list[str], eta: float, eps: float = 0.001) -> None:
        self._experts = experts
        self._probs = ones(len(experts), dtype=float64) / self.n_experts
        self._cumulative_losses = zeros(len(experts), dtype=float64)

        # self._eta = eta

        self._t = 1
        self._f = self.n_experts // 3
        self._eps = eps
        self._eps_t = self._eps

        self._max_outcome: float = 100.0
        self._min_outcome: float = -100.0

    @property
    def probs(self) -> dict[str, float]:
        return {j: self._probs[i] for i, j in enumerate(self._experts)}

    @property
    def n_experts(self) -> int:
        return len(self._experts)

    @property
    def _eta(self) -> float:
        return sqrt(log(self.n_experts) / self._t)

    def _update_probs(self, outcomes: dict[str, NDArray[float64]]) -> None:
        losses = array([loss_func(j, outcomes) for j in self._experts], dtype=float64)
        max_loss = max(losses.max(), 1.0)
        normalized_losses = losses / max_loss
        self._cumulative_losses += normalized_losses

        weights = exp(-self._eta * self._cumulative_losses)
        self._probs = weights / weights.sum()

    def make_prediction(
        self, outcomes: dict[str, NDArray[float64]]
    ) -> NDArray[float64]:
        self._update_probs(outcomes)

        outcomes_ = stack([outcomes[j] for j in self._experts], dtype=float64)

        # Probabilistic selection
        # idx = choice(self.n_experts, p=self._probs)

        # return outcomes_[idx]

        # Weighted average
        # return average(outcomes_, axis=0, weights=self._probs)

        # Probability-trimmed weighting
        # min_f_prob = min_f(self._probs, f=self._f)
        # if min_f_prob is None:
        #     return average(outcomes_, axis=0)

        # weights = where(self._probs > min_f_prob, 1.0, self._eps_t)

        # self._t += 1
        # self._eps_t *= self._eps

        # return average(outcomes_, axis=0, weights=weights)

        # Bounded weighted average
        outcomes_ = maximum(outcomes_, self._min_outcome)
        outcomes_ = minimum(outcomes_, self._max_outcome)

        return average(outcomes_, axis=0, weights=self._probs)
