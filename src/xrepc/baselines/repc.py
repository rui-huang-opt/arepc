import numpy as np
from numpy.typing import NDArray
from numpy.linalg import norm


def min_f(values: NDArray[np.float64], f: int) -> float | None:
    """
    The min-f value is defined as the f-th smallest unique value in the array,
    but it is not allowed to be the largest value.
    If there is only one unique value, return None to indicate that all values are equal.
    """

    if values.size == 0:
        raise ValueError("Probabilities array is empty.")

    deduped_values = np.unique(values)

    if len(deduped_values) == 1:
        return None

    if f >= len(deduped_values):
        return deduped_values[-2]
    else:
        return deduped_values[f - 1]


class RepC:
    """
    Reputation-based Consensus (RepC) baseline implementation.

    The paper introducing the RepC algorithm is:
    Ramos, G., Silvestre, D., & Silvestre, C. (2023).
    "A Discrete-Time Reputation-Based Resilient Consensus Algorithm for Synchronous or Asynchronous Communications".
    IEEE Transactions on Automatic Control, 69(1), 543-550.

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
        self._probs: NDArray[np.float64] = np.zeros(len(neighbors))
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
        return {j: self._probs[i] for i, j in enumerate(self._neighbors)}

    def _compute_reputation(
        self, neighbor_states: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        """
        Compute reputation scores for each neighbor from their state vectors.

        For each neighbor j with state x_j, we first compute a per-neighbor
        loss as the mean pairwise L2 distance to all neighbors:

            loss_j = (1 / |N_i|) * sum_{k in N_i} ||x_j - x_k||_2

        The reputation is then defined as:

            r_j = 1 - loss_j

        Note: the aggregation used to form `loss_j` mirrors the one used in the quasi-geometric-median-style loss in `xrepc.loss_func.py`.
        We re-implement it locally here (instead of importing/calling the shared function) to keep this benchmark code self-contained.
        """

        differences = neighbor_states[:, None, :] - neighbor_states[None, :, :]
        distances = norm(differences, axis=2)
        losses = np.mean(distances, axis=1)
        return 1.0 - losses

    def _normalizer(self, reputations: NDArray[np.float64]) -> NDArray[np.float64]:
        """
        Normalize reputations to probabilities using the RepC normalization scheme.
        """

        min_f_reputation = min_f(reputations, f=self._f)

        if min_f_reputation is None:
            reputations.fill(1.0)
        else:
            max_reputation: float = reputations.max()
            reputation_range = max_reputation - min_f_reputation
            reputations -= min_f_reputation
            reputations /= reputation_range
            reputations[reputations <= 0] = self._eps_t

        self._eps_t *= self._eps

        return reputations / reputations.sum()

    def aggregate(
        self,
        local_state: NDArray[np.float64],
        neighbor_states: dict[str, NDArray[np.float64]],
    ) -> NDArray[np.float64]:
        neighbor_states_ = np.vstack([neighbor_states[j] for j in self._neighbors])

        reputations = self._compute_reputation(neighbor_states_)
        self._probs = self._normalizer(reputations)

        estimate = self._probs @ neighbor_states_

        return local_state * (1 - self._alpha) + estimate * self._alpha
