from numpy import float64, mean, zeros_like
from numpy.typing import NDArray
from numpy.linalg import norm
from ..utils import min_f


# 考虑到后续可能会有其它的惩罚函数，因此不把这个函数写在utils里
def penalty_func(j: str, x_js: dict[str, NDArray[float64]]) -> float:
    return mean([norm(x_js[j] - x_js[k]) for k in x_js], dtype=float64)


class RepC:
    """Reputation-based Consensus (RepC) baseline implementation."""

    def __init__(self, neighbors: list[str], eps: float = 0.001) -> None:
        self._reputation_scores = {j: 1.0 for j in neighbors}
        self._f = len(neighbors) // 3
        self._eps = eps
        self._confidence = eps

    @property
    def f(self) -> int:
        return self._f

    def _reputation_update(self, neighbor_states: dict[str, NDArray[float64]]) -> None:
        for j in self._reputation_scores:
            penalty = penalty_func(j, neighbor_states)
            self._reputation_scores[j] = 1.0 - penalty

    def _reputation_normalization(self) -> None:
        min_f_score = min_f(self._reputation_scores, f=self._f)
        max_score = max(self._reputation_scores.values())

        for j in self._reputation_scores:
            score = self._reputation_scores[j]

            if min_f_score is None:
                normalized_score = 1.0
            else:
                normalized_score = (score - min_f_score) / (max_score - min_f_score)

            if normalized_score > 0:
                self._reputation_scores[j] = normalized_score
            else:
                self._reputation_scores[j] = self._confidence

        self._confidence *= self._eps

    def update(
        self,
        local_state: NDArray[float64],
        neighbor_states: dict[str, NDArray[float64]],
    ) -> NDArray[float64]:
        self._reputation_update(neighbor_states)
        self._reputation_normalization()

        weighted_sum = sum(
            [self._reputation_scores[j] * neighbor_states[j] for j in neighbor_states],
            start=zeros_like(local_state),
        )
        total_weight = sum(self._reputation_scores.values())

        return (local_state + weighted_sum) / (1.0 + total_weight)
