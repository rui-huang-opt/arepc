from numpy import float64, mean, ones, array, where, stack, average, append
from numpy.typing import NDArray
from numpy.linalg import norm
from ..utils import min_f


# 考虑到后续可能会有其它的惩罚函数，因此不把这个函数写在utils里
def loss_func(j: str, x_js: dict[str, NDArray[float64]]) -> float:
    return mean([norm(x_js[j] - x_js[k]) for k in x_js], dtype=float64)


class RepC:
    """Reputation-based Consensus (RepC) baseline implementation."""

    def __init__(self, neighbors: list[str], eps: float = 0.001) -> None:
        self._neighbors = neighbors
        self._scores = ones(len(neighbors), dtype=float64)
        self._f = len(neighbors) // 3
        self._eps = eps
        self._confidence = eps

    def _reputation_update(self, neighbor_states: dict[str, NDArray[float64]]) -> None:
        losses = array([loss_func(j, neighbor_states) for j in self._neighbors])
        self._scores = 1.0 - losses

    def _reputation_normalization(self) -> None:
        min_f_score = min_f(self._scores, f=self._f)

        if min_f_score is None:
            self._scores = ones(len(self._neighbors))
        else:
            max_score = self._scores.max()
            new_scores = (self._scores - min_f_score) / (max_score - min_f_score)
            self._scores = where(new_scores > 0, new_scores, self._confidence)

        self._confidence *= self._eps

    def aggregate_states(
        self,
        local_state: NDArray[float64],
        neighbor_states: dict[str, NDArray[float64]],
    ) -> NDArray[float64]:
        self._reputation_update(neighbor_states)
        self._reputation_normalization()

        states = stack([neighbor_states[j] for j in self._neighbors] + [local_state])
        weights = append(self._scores, 1.0)

        return average(states, axis=0, weights=weights)
