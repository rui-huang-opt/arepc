from numpy import float64
from numpy import mean, where, vstack, unique
from numpy import ones
from numpy.typing import NDArray
from numpy.linalg import norm


def min_f(values: NDArray[float64], f: int) -> float | None:
    """
    The min-f value is defined as the f-th smallest unique value in the array,
    but it is not allowed to be the largest value.
    If there is only one unique value, return None to indicate that all values are equal.
    """

    if values.size == 0:
        raise ValueError("Probabilities array is empty.")

    deduped_values = unique(values)

    if len(deduped_values) == 1:
        return None

    if f >= len(deduped_values):
        return deduped_values[-2]
    else:
        return deduped_values[f - 1]


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
        self._reputations = ones(len(neighbors))
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
        probs = self._reputations / self._reputations.sum()
        return {j: probs[i] for i, j in enumerate(self._neighbors)}

    def _update_reputation(self, neighbor_states: NDArray[float64]) -> None:
        differences = neighbor_states[:, None, :] - neighbor_states[None, :, :]
        distances = norm(differences, axis=2)
        losses = mean(distances, axis=1)
        self._reputations = 1.0 - losses

    def _normalize_reputation(self) -> None:
        min_f_reputation = min_f(self._reputations, f=self._f)

        if min_f_reputation is None:
            self._reputations.fill(1.0)
        else:
            max_reputation = self._reputations.max()
            reputation_range = max_reputation - min_f_reputation
            new_reputations = (self._reputations - min_f_reputation) / reputation_range
            self._reputations = where(new_reputations > 0, new_reputations, self._eps_t)

        self._eps_t *= self._eps

    def aggregate(
        self,
        local_state: NDArray[float64],
        neighbor_states: dict[str, NDArray[float64]],
    ) -> NDArray[float64]:
        neighbor_states_ = vstack([neighbor_states[j] for j in self._neighbors])

        self._update_reputation(neighbor_states_)
        self._normalize_reputation()

        probs = self._reputations / self._reputations.sum()

        estimate = probs @ neighbor_states_

        return local_state * (1 - self._alpha) + estimate * self._alpha
