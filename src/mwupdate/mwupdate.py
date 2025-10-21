from numpy import float64, mean, exp, sqrt, log, average
from numpy.linalg import norm
from numpy.typing import NDArray
from .utils import min_f


# 考虑到后续可能会有其它的惩罚函数，因此不把这个函数写在utils里
def penalty_func(j: str, x_js: dict[str, NDArray[float64]]) -> float:
    return mean([norm(x_js[j] - x_js[k]) for k in x_js], dtype=float64)


class MWUpdate:
    def __init__(self, experts: list[str], eps: float = 0.001) -> None:
        self._n_experts = len(experts)
        self._probs = {j: 1.0 / self._n_experts for j in experts}
        self._log_weights = {j: 0.0 for j in experts}
        self._t = 1
        self._f = self._n_experts // 3
        self._eps = eps
        self._eps_t = self._eps

    @property
    def probs(self) -> dict[str, float]:
        return self._probs

    @property
    def _eta(self) -> float:
        return sqrt(log(self._n_experts) / self._t)

    def _update_probs(self, outcomes: dict[str, NDArray[float64]]) -> None:
        penalties = {j: penalty_func(j, outcomes) for j in outcomes}
        max_penalty = max(max(penalties.values()), 1.0)

        for j in self._log_weights:
            self._log_weights[j] += -self._eta * penalties[j] / max_penalty

        max_log_weight = max(self._log_weights.values())

        for j in self._log_weights:
            self._log_weights[j] -= max_log_weight

        total_weight: float = sum(exp(self._log_weights[j]) for j in self._log_weights)

        for j in self._probs:
            self._probs[j] = exp(self._log_weights[j]) / total_weight

    def make_prediction(
        self, outcomes: dict[str, NDArray[float64]]
    ) -> NDArray[float64]:
        self._update_probs(outcomes)

        min_f_prob = min_f(self._probs, f=self._f)
        if min_f_prob is None:
            return average([outcomes[j] for j in outcomes], axis=0)

        outcomes_ = [outcomes[j] for j in self._probs]
        weights = [1.0 if p > min_f_prob else self._eps_t for p in self._probs.values()]

        self._t += 1
        self._eps_t *= self._eps

        return average(outcomes_, axis=0, weights=weights)
