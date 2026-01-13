import logging
from dataclasses import dataclass
from typing import Collection

import numpy as np
from numpy.typing import NDArray
from scipy.spatial.distance import cdist

from ..network import NetworkOps

logger = logging.getLogger(__name__)


def min_f(values: NDArray[np.float64], f: int) -> float | None:
    """
    The min-f value is defined as the f-th smallest unique value in the array,
    but it is not allowed to be the largest value.
    If there is only one unique value, return None to indicate that all values are equal.

    Args:
        values (NDArray[np.float64]): Array of values.

        f (int): The rank of the value to retrieve (1-based index).

    Returns:
        float | None: The f-th smallest unique value, or None if all values are equal
    """
    deduped_values = np.unique(values)

    if len(deduped_values) == 1 or f == 0:
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
    value : NDArray[np.float64]
        Array of reputation scores for each neighbor.
    """

    value: NDArray[np.float64]

    def update(self, neighbor_states: NDArray[np.float64]) -> None:
        """
        Compute reputation scores for each neighbor from their state vectors.

        For each neighbor j with state x_j, we first compute a per-neighbor mean pairwise distance:

            mean_pairwise_distance_j = (1 / |N_i|) * sum_{k in N_i} ||x_j - x_k||_2

        The reputation is then defined as:

            r_j = 1 - mean_pairwise_distance_j

        Args:
            neighbor_states (NDArray[np.float64]): 2D array where each row corresponds to a neighbor's state vector.
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

        Args:
            f (int): Number of tolerated faulty nodes.

            eps (float): Confidence threshold for reputations.
        """
        min_f_value = min_f(self.value, f=f)

        if min_f_value is None:
            self.value.fill(1.0 / len(self.value))
        else:
            max_reputation: float = self.value.max()
            reputation_range = max_reputation - min_f_value
            self.value -= min_f_value
            self.value /= reputation_range
            self.value[self.value < 0.0] = eps
            self.value /= self.value.sum()

    def to_dict(self, neighbor_names: Collection[str]) -> dict[str, float]:
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

    f : int
        The number of tolerated faulty nodes.

    eps : float, optional
        Confidence threshold for reputation normalization. Defaults to 0.001.

    Attributes
    ----------
    probs : dict[str, float]
        Current probabilities assigned to each neighbor.
    """

    def __init__(
        self,
        ops: NetworkOps,
        alpha: float,
        f: int,
        eps: float = 0.001,
    ) -> None:
        self._ops = ops
        self.alpha = alpha

        if not (0.0 <= eps < 1.0):
            err_msg = "eps must be in the range [0, 1)."
            logger.error(err_msg)
            raise ValueError(err_msg)

        self._eps = eps
        self._eps_t = eps

        if f < 0:
            err_msg = "f must be a non-negative integer."
            logger.error(err_msg)
            raise ValueError(err_msg)

        self._f = f

        self._reputations = Reputations(np.zeros(ops.degree, dtype=np.float64))

    @property
    def reputations(self) -> dict[str, float]:
        return self._reputations.to_dict(self._ops.neighbors)

    def weighted_mix(self, local_state: NDArray[np.float64]) -> NDArray[np.float64]:
        """
        The operator that aggregates neighbor states with local state using RepC.

        Args:
            local_state (NDArray[np.float64]): The local state array.

        Returns:
            NDArray[np.float64]: The updated local state array.
        """
        neighbor_states = self._ops.exchange_as_array(local_state)

        self._reputations.update(neighbor_states)
        self._reputations.normalize(f=self._f, eps=self._eps_t)

        self._eps_t *= self._eps

        center_proxy = self._reputations.value @ neighbor_states

        return local_state * (1 - self.alpha) + center_proxy * self.alpha
