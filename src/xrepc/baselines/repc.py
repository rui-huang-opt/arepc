import numpy as np
from numpy.typing import NDArray

from ..network import NetworkOps
from ..loss_func import quasi_geometric_median_loss


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
        self._probs = np.zeros(ops.num_neighbors, dtype=np.float64)
        self._f = f
        self._alpha = alpha

        if not (0.0 < eps < 1.0):
            raise ValueError("eps must be in the range (0, 1).")

        self._eps = eps
        self._eps_t = eps

    @property
    def probs(self) -> dict[str, float]:
        return {j: self._probs[i] for i, j in enumerate(self._ops.neighbor_names)}

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

        Note: the loss function used here is adopted in the proposed XRepC method.
        And is named "quasi-geometric median loss" in our implementation.
        """

        losses = quasi_geometric_median_loss(neighbor_states)
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

    def step(self, local_state: NDArray[np.float64]) -> NDArray[np.float64]:
        neighbor_state_map = self._ops.exchange(local_state)
        neighbor_states = np.array(list(neighbor_state_map.values()))

        reputations = self._compute_reputation(neighbor_states)
        self._probs = self._normalizer(reputations)

        neighbor_estimate = self._probs @ neighbor_states

        return local_state * (1 - self._alpha) + neighbor_estimate * self._alpha
