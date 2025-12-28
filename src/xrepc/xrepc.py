from typing import Sequence

import numpy as np
from numpy.typing import NDArray

from .loss_func import LOSS_FUNC_MAP
from .selector import SELECTOR_MAP


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

    loss_func : str, optional
        Loss function to evaluate neighbor predictions.
        Options are
        "gm" (geometric median loss),
        "qgm" (quasi-geometric median loss),
        "cwm" (coordinate-wise median loss),
        and "mean" (mean loss).
        Defaults to "cwm".

    decay : float, optional
        Decay factor for past losses, must be in [0, 1]. Defaults to 0.3.

    selector : str, optional
        Selector function to convert losses to probabilities.
        Options are "softmax", "sparsemax", and "entmax15".
        Defaults to "sparsemax".

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
        loss_func: str = "cwm",
        decay: float = 0.8,
        selector: str = "sparsemax",
    ) -> None:
        super().__init__()

        self._neighbors = neighbors
        self._alpha = alpha
        self._eta = eta
        self._loss_func = LOSS_FUNC_MAP[loss_func]

        if not (0.0 <= decay <= 1.0):
            raise ValueError("decay factor must be in [0, 1].")

        self._decay = decay
        self._selector = SELECTOR_MAP[selector]

        self._total_expert_losses = np.zeros(self.n_neighbors, dtype=np.float64)
        self._probs = np.zeros(self.n_neighbors, dtype=np.float64)

    @property
    def n_neighbors(self) -> int:
        return len(self._neighbors)

    @property
    def probs(self) -> dict[str, float]:
        return {j: self._probs[i] for i, j in enumerate(self._neighbors)}

    def aggregate(
        self,
        local_state: NDArray[np.float64],
        neighbor_states: dict[str, NDArray[np.float64]],
    ) -> NDArray[np.float64]:
        advices = np.stack([neighbor_states[j] for j in self._neighbors])

        expert_losses = self._loss_func(advices)

        self._total_expert_losses *= self._decay
        self._total_expert_losses += expert_losses

        self._probs = self._selector(-self._eta * self._total_expert_losses)

        prediction = self._probs @ advices

        return local_state * (1 - self._alpha) + prediction * self._alpha
