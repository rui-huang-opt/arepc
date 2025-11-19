from numpy import float64
from numpy import mean, where, vstack, unique
from numpy import ones
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
    """
    Reputation-based Consensus (RepC) baseline implementation.

    Parameters
    ----------
    neighbors : list[str]
        List of neighbor identifiers.

    alpha : float
        Mixing parameter between local state and aggregated prediction.

    eps : float, optional
        Confidence threshold for reputation normalization. Defaults to 0.001.

    f : int, optional
        Number of tolerated faulty nodes. Defaults to 1.

    Attributes
    ----------
    n_neighbors : int
        Number of neighbors.

    probs : dict[str, float]
        Current probabilities assigned to each neighbor.

    Methods
    -------
    aggregate(local_state: NDArray[float64], neighbor_states: dict[str, NDArray[float64]]) -> NDArray[float64]
        Aggregates neighbor states with the local state using the RepC algorithm.
    """

    def __init__(
        self, neighbors: list[str], alpha: float, eps: float = 0.001, f: int = 1
    ) -> None:
        self._neighbors = neighbors
        self._scores = ones(len(neighbors), dtype=float64)
        self._f = f
        self._alpha = alpha

        if not (0.0 < eps < 1.0):
            raise ValueError("eps must be in the range (0, 1).")

        self._eps = eps
        self._eps_t = eps

    @property
    def n_neighbors(self) -> int:
        return len(self._neighbors)

    @property
    def probs(self) -> dict[str, float]:
        probs = self._scores / self._scores.sum()
        return {j: probs[i] for i, j in enumerate(self._neighbors)}

    def _reputation_update(self, neighbor_states: NDArray[float64]) -> None:
        differences = neighbor_states[:, None, :] - neighbor_states[None, :, :]
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
            self._scores = where(new_scores > 0, new_scores, self._eps_t)

        self._eps_t *= self._eps

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
