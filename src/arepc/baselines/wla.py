import numpy as np
from numpy.typing import NDArray
import numpy.linalg as npl

from ..network import NetworkOps
from ..normalizer import softmax


class QCons:
    """
    Q-consensus baseline implementation.

    The paper introducing Q-consensus is:
    Hou, J., Wang, F., Wang, L., & Chen, Z. (2021).
    "Reinforcement learning based multi-agent resilient control: From deep neural networks to an adaptive law".
    Proceedings of the AAAI Conference on Artificial Intelligence, 35(9), 7737-7745.
    """

    def __init__(
        self, ops: NetworkOps, beta: float, eta: float, alpha: float | None = None
    ) -> None:
        self._ops = ops

        self.beta = beta
        self.eta = eta
        self.alpha = 1 - 1 / (self._ops.degree + 1) if alpha is None else alpha

        self._q_values = np.ones(self._ops.degree)
        self._weights = np.zeros(self._ops.degree)

    @property
    def weights(self) -> dict[str, float]:
        return {j: float(self._weights[i]) for i, j in enumerate(self._ops.neighbors)}

    def weighted_mix(self, local_state: NDArray[np.float64]) -> NDArray[np.float64]:
        """
        The operator that aggregates neighbor states with local state using Q-consensus.

        Args:
            local_state (NDArray[np.float64]): The local state array.

        Returns:
            NDArray[np.float64]: The computed Q-consensus value.
        """
        neighbor_states = self._ops.exchange_as_array(local_state)

        dists: NDArray[np.float64] = npl.norm(neighbor_states - local_state, axis=1)
        rewards = np.exp(-dists * self.beta)
        self._q_values = self._q_values + self.eta * (rewards - self._q_values)

        self._weights = self._q_values / np.sum(self._q_values)
        center_proxy = self._weights @ neighbor_states

        return local_state * (1 - self.alpha) + center_proxy * self.alpha


class WLA:
    """
    Weighted Learning Algorithm (WLA) baseline implementation.
    It is modified from QCons, i.e., updating the Q-values by multiplying the reward by a learning rate eta rather than adding it.

    This implementation can be regarded as a special case of our family of algorithms with a specific choice:
    loss: local state loss (norm of the difference between neighbor states and local state);
    accumulation: full-history sum;
    normalization: softmax.
    Although it can be cast within our general framework, we implement it separately as it does not fully align with our design principles:
    (i) the loss is not constructed solely from neighbors' states rather than being locally generated, and
    (ii) the accumulation mechanism lacks any forgetting mechanism.

    This implementation is imtroduced in the paper:
    Hou, J., Chen, Z., Lin, Z., Wei, C., Zheng, J., Wang, F., Xiang, M., & Xie, Y. (2023).
    "Resilient consensus via weight learning and its application in fault-tolerant clock synchronization".
    IEEE Transactions on Control of Network Systems, 10(4), 2097-2107.
    """

    def __init__(
        self,
        ops: NetworkOps,
        eta: float,
        alpha: float | None = None,
    ) -> None:
        self._ops = ops

        self.eta = eta
        self.alpha = 1 - 1 / (self._ops.degree + 1) if alpha is None else alpha

        self._cumulative_losses = np.zeros(self._ops.degree)
        self._reputations = np.zeros(self._ops.degree)

    @property
    def reputations(self) -> dict[str, float]:
        return {
            j: float(self._reputations[i]) for i, j in enumerate(self._ops.neighbors)
        }

    def weighted_mix(self, local_state: NDArray[np.float64]) -> NDArray[np.float64]:
        """
        The operator that aggregates neighbor states with local state using Q-consensus.

        Args:
            local_state (NDArray[np.float64]): The local state array.

        Returns:
            NDArray[np.float64]: The computed Q-consensus value.
        """
        neighbor_states = self._ops.exchange_as_array(local_state)

        current_losses = npl.norm(neighbor_states - local_state, axis=1)
        self._cumulative_losses += current_losses
        self._reputations = softmax(-self.eta * self._cumulative_losses)

        center_proxy = self._reputations @ neighbor_states

        return local_state * (1 - self.alpha) + center_proxy * self.alpha
