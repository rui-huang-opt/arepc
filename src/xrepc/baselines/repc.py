from dataclasses import dataclass
from typing import Collection

import numpy as np
from numpy.typing import NDArray
from scipy.spatial.distance import cdist

from ..network import NetworkOps


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


@dataclass(slots=True)
class Reputations:
    """
    Data class to hold reputations of neighbor nodes.

    Parameters
    ----------
    names : list[str]
        List of neighbor names.

    scores : NDArray[np.float64]
        Array of reputation scores corresponding to each neighbor.
    """

    value: NDArray[np.float64] = np.array([], dtype=np.float64)

    def update(self, neighbor_states: NDArray[np.float64]) -> None:
        """
        Compute reputation scores for each neighbor from their state vectors.

        For each neighbor j with state x_j, we first compute a per-neighbor mean pairwise distance:

            mean_pairwise_distance_j = (1 / |N_i|) * sum_{k in N_i} ||x_j - x_k||_2

        The reputation is then defined as:

            r_j = 1 - mean_pairwise_distance_j
        """
        pairwise_distances = cdist(neighbor_states, neighbor_states, metric="euclidean")
        self.value = 1.0 - pairwise_distances.mean(axis=1)

    def normalize(self, f: int, eps: float) -> None:
        """
        Normalize reputations to probabilities using the RepC normalization scheme.

        The normalization consists of two main steps:
        1. Shift and scale reputations so that the worst f reputations are set to a small threshold eps_t,
           and all others are adjusted accordingly.
        2. Normalize the adjusted reputations to sum to 1.
        """
        min_f_reputation = min_f(self.value, f=f)

        if min_f_reputation is None:
            self.value.fill(1.0 / len(self.value))
        else:
            max_reputation: float = self.value.max()
            reputation_range = max_reputation - min_f_reputation
            self.value -= min_f_reputation
            self.value /= reputation_range
            self.value[self.value < 0.0] = eps
            self.value /= self.value.sum()

    def to_dict(self, neighbor_names: Collection[str]) -> dict[str, float]:
        if self.value.size == 0:
            raise ValueError("Reputations have not been computed yet.")
        return {j: self.value[i] for i, j in enumerate(neighbor_names)}


class RepC:
    """
    Reputation-based Consensus (RepC) baseline implementation.

    The paper introducing the RepC algorithm is:
    Ramos, G., Silvestre, D., & Silvestre, C. (2023).
    "A Discrete-Time Reputation-Based Resilient Consensus Algorithm for Synchronous or Asynchronous Communications".
    IEEE Transactions on Automatic Control, 69(1), 543-550.

    Parameters
    ----------
    ops : NetworkOps
        Network operations handler implementing the 'NetworkOps' protocol (defined in the 'network' module).

    alpha : float
        Mixing parameter between local state and aggregated prediction.

    eps : float, optional
        Confidence threshold for reputation normalization. Defaults to 0.001.

    f : int, optional
        Number of tolerated faulty nodes. Defaults to 1.

    Attributes
    ----------
    probs : dict[str, float]
        Current probabilities assigned to each neighbor.
    """

    def __init__(
        self, ops: NetworkOps, alpha: float, eps: float = 0.001, f: int = 1
    ) -> None:
        self._ops = ops
        self._f = f
        self._alpha = alpha

        if not (0.0 < eps < 1.0):
            raise ValueError("eps must be in the range (0, 1).")

        self._eps = eps
        self._eps_t = eps

        # Since reputations are computed at each step, we initialize an empty array.
        self._reputations = Reputations()

    @property
    def reputations(self) -> dict[str, float]:
        return self._reputations.to_dict(self._ops.neighbor_names)

    def step(self, local_state: NDArray[np.float64]) -> NDArray[np.float64]:
        neighbor_state_map = self._ops.exchange(local_state)
        neighbor_states = np.array(list(neighbor_state_map.values()))

        self._reputations.update(neighbor_states)
        self._reputations.normalize(f=self._f, eps=self._eps_t)

        self._eps_t *= self._eps

        neighbor_estimate = self._reputations.value @ neighbor_states

        return local_state * (1 - self._alpha) + neighbor_estimate * self._alpha
