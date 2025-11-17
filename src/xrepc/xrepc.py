from typing import Sequence
from numpy import float64, stack, zeros
from numpy.typing import NDArray
from .loss_func import LossFunc, CoordinateWiseMedianLoss
from .utils import softmax


class XRepC:
    """
    Exponential Replicator Dynamics for Collaborative Learning (XRepC).

    Parameters
    ----------
    neighbors : Sequence[str]
        List of neighbor identifiers.

    alpha : float
        Mixing parameter between local state and aggregated prediction.

    eta : float
        Learning rate for updating probabilities.

    loss_func : LossFunc, optional
        Loss function to evaluate neighbor outcomes. Defaults to CoordinateWiseMedianLoss.

    decay : float, optional
        Decay factor for past losses, must be in [0, 1]. Defaults to 0.3.

    Attributes
    ----------
    n_neighbors : int
        Number of neighbors.

    probs : dict[str, float]
        Current probabilities assigned to each neighbor.

    Methods
    -------
    aggregate(local_state: NDArray[float64], neighbor_states: dict[str, NDArray[float64]]) -> NDArray[float64]
        Aggregates neighbor states with the local state using the XRepC algorithm.
    """

    def __init__(
        self,
        neighbors: Sequence[str],
        alpha: float,
        eta: float,
        loss_func: LossFunc = CoordinateWiseMedianLoss(),
        decay: float = 0.3,
    ) -> None:
        super().__init__()

        self._neighbors = neighbors
        self._alpha = alpha
        self._eta = eta
        self._loss_func = loss_func

        if not (0.0 <= decay <= 1.0):
            raise ValueError("decay factor must be in [0, 1].")

        self._decay = decay

        self._losses_memory = zeros(self.n_neighbors, dtype=float64)
        self._probs = zeros(self.n_neighbors, dtype=float64)

    @property
    def n_neighbors(self) -> int:
        return len(self._neighbors)

    @property
    def probs(self) -> dict[str, float]:
        return {j: self._probs[i] for i, j in enumerate(self._neighbors)}

    def _update_probs(self, outcomes: NDArray[float64]) -> None:
        losses = self._loss_func(outcomes)

        self._losses_memory *= self._decay
        self._losses_memory += losses

        self._probs = softmax(-self._eta * self._losses_memory)

    def aggregate(
        self,
        local_state: NDArray[float64],
        neighbor_states: dict[str, NDArray[float64]],
    ) -> NDArray[float64]:
        outcomes = stack([neighbor_states[j] for j in self._neighbors])
        self._update_probs(outcomes)
        prediction = self._probs @ outcomes

        return local_state * (1 - self._alpha) + prediction * self._alpha
