import numpy as np
from numpy.typing import NDArray
import numpy.linalg as npl

from ..network import NetworkOps


class QCons:
    """
    Q-consensus baseline implementation.
    """

    def __init__(
        self, ops: NetworkOps, beta: float, eta: float, alpha: float | None = None
    ) -> None:
        self._ops = ops
        self._beta = beta
        self._eta = eta
        self._alpha = 1 - 1 / (self._ops.degree + 1) if alpha is None else alpha
        self._q_values = np.ones(self._ops.degree)
        self._weights = np.zeros(self._ops.degree)

    @property
    def weights(self) -> dict[str, float]:
        return {j: float(self._weights[i]) for i, j in enumerate(self._ops.neighbors)}

    def weighted_mix(self, local_state: NDArray[np.float64]) -> NDArray[np.float64]:
        """
        Compute the Q-consensus value by applying the min-f function to the neighbor values.

        Args:
            local_state (NDArray[np.float64]): The local state array.
        Returns:
            NDArray[np.float64]: The computed Q-consensus value.
        """
        neighbor_states = self._ops.exchange_as_array(local_state)

        dists: NDArray[np.float64] = npl.norm(neighbor_states - local_state, axis=1)
        rewards = np.exp(-dists * self._beta)
        self._q_values = self._q_values + self._eta * (rewards - self._q_values)

        self._weights = self._q_values / np.sum(self._q_values)
        center_proxy = self._weights @ neighbor_states

        return local_state * (1 - self._alpha) + center_proxy * self._alpha
