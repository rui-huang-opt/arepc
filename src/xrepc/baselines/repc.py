from numpy import float64
from numpy import mean, where, vstack, average, append, unique
from numpy import ones, newaxis
from numpy.typing import NDArray
from numpy.linalg import norm


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


class RepC:
    """Reputation-based Consensus (RepC) baseline implementation."""

    def __init__(self, neighbors: list[str], alpha: float, eps: float = 0.001) -> None:
        self._neighbors = neighbors
        self._scores = ones(len(neighbors), dtype=float64)
        self._f = len(neighbors) // 3
        self._alpha = alpha
        self._eps = eps
        self._confidence = eps

    @property
    def n_neighbors(self) -> int:
        return len(self._neighbors)

    @property
    def probs(self) -> dict[str, float]:
        probs = self._scores / self._scores.sum()
        return {j: probs[i] for i, j in enumerate(self._neighbors)}

    def _reputation_update(self, neighbor_states: NDArray[float64]) -> None:
        # differences shape: (n_neighbors, n_neighbors, state_dim)
        # Compute pairwise differences between neighbor states
        differences = neighbor_states[:, newaxis, :] - neighbor_states[newaxis, :, :]
        distances = norm(differences, axis=2)
        losses = mean(distances, axis=1)
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

    def aggregate(
        self,
        local_state: NDArray[float64],
        neighbor_states: dict[str, NDArray[float64]],
    ) -> NDArray[float64]:
        neighbor_states_ = vstack([neighbor_states[j] for j in self._neighbors])

        self._reputation_update(neighbor_states_)
        self._reputation_normalization()

        probs = self._scores / self._scores.sum()
        estimate = probs @ neighbor_states_

        return (1 - self._alpha) * local_state + self._alpha * estimate
