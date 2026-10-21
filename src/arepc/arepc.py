from typing import Literal, Collection

import numpy as np
from numpy.typing import NDArray

from .loss_func import LossFunc, LOSS_FUNC_MAP
from .accumulator import make_accumulator
from .normalizer import NORMALIZER_MAP


class ARepC:
    """
    Adaptive Reputation-based Consensus (ARepC) robust aggregation algorithm.

    ARepC evaluates neighbor states using a loss function, accumulates
    the losses over time, and converts them into reputation weights.
    These weights are then used to aggregate neighbor states with the
    local state.

    Network communication is handled externally. ARepC only performs
    reputation updates and state aggregation.

    Parameters
    ----------
    neighbors : Collection[str]
        Identifiers of neighboring nodes. Internally sorted to maintain
        a consistent ordering of neighbor states and reputations.

    alpha : float
        Mixing parameter controlling the contribution of aggregated
        neighbor states relative to the local state.

    eta : float
        Temperature parameter controlling the sensitivity of
        reputation weights to accumulated losses.

    loss_func : LossFunc | {"cmed", "qmed", "gmed", "mean"}, optional
        Loss function used to evaluate neighbor states.

        Available predefined options are:

        - "cmed": Coordinate-wise median loss.
        - "qmed": Quasi-geometric median loss.
        - "gmed": Geometric median loss.
        - "mean": Mean loss.

        Defaults to "cmed".

        Custom loss functions are also supported. A loss function must
        accept a 2D NumPy array of stacked neighbor states and return
        a 1D array containing one loss value per neighbor.

        Configurable loss functions, such as GeometricMedianLoss and
        TrimmedMeanLoss, can be instantiated separately and passed
        through this parameter.

    accumulation : {"exp-decay", "moving-horizon"}, optional
        Method for accumulating historical losses.

        - "exp-decay": Exponentially decayed accumulation.
        - "moving-horizon": Accumulation over a finite moving window.

        Defaults to "exp-decay".

    normalization : {"softmax", "sparsemax", "1.5-entmax"}, optional
        Normalization function used to convert accumulated losses
        into reputation weights. Defaults to "softmax".

    horizon : float, optional
        Parameter controlling the accumulation method.

        For "exp-decay", the decay coefficient is 1 - 1 / horizon.
        For "moving-horizon", the window size is int(horizon).

        Defaults to 5.0.

    Attributes
    ----------
    reputations : dict[str, float]
        Current reputation weights assigned to neighboring nodes.
    """

    def __init__(
        self,
        neighbors: Collection[str],
        alpha: float,
        eta: float,
        loss_func: LossFunc | Literal["cmed", "qmed", "gmed", "mean"] = "cmed",
        accumulation: Literal["exp-decay", "moving-horizon"] = "exp-decay",
        normalization: Literal["softmax", "sparsemax", "1.5-entmax"] = "softmax",
        horizon: float = 5.0,
    ) -> None:
        self._neighbors = tuple(sorted(neighbors))
        self.alpha = alpha
        self.eta = eta

        if isinstance(loss_func, str):
            self._loss_func = LOSS_FUNC_MAP[loss_func]
        else:
            self._loss_func = loss_func

        degree = len(neighbors)

        self._accumulator = make_accumulator(accumulation, horizon, dim=degree)
        self._normalizer = NORMALIZER_MAP[normalization]

        self._reputations = np.zeros(degree, dtype=np.float64)

    @property
    def reputations(self) -> dict[str, float]:
        return {j: float(self._reputations[i]) for i, j in enumerate(self._neighbors)}

    def mix(
        self,
        local_state: NDArray[np.float64],
        neighbor_states: dict[str, NDArray[np.float64]],
    ) -> NDArray[np.float64]:
        """
        Update neighbor reputations and aggregate states using ARepC.

        Neighbor losses are evaluated and accumulated to update reputation
        weights. The weighted combination of neighbor states is then mixed
        with the local state:

            mixed_state = (1 - alpha) * local_state + alpha * sum_j(reputation_j * neighbor_state_j)

        Parameters
        ----------
        local_state : NDArray[np.float64]
            Current local state array.

        neighbor_states : dict[str, NDArray[np.float64]]
            Mapping from neighbor IDs to their state arrays. Each
            configured neighbor must have a corresponding state.

        Returns
        -------
        NDArray[np.float64]
            Aggregated state obtained using the updated reputations.
        """
        stacked_states = np.stack([neighbor_states[j] for j in self._neighbors], axis=0)
        current_losses = self._loss_func(stacked_states)
        cumulative_losses = self._accumulator(current_losses)
        self._reputations = self._normalizer(-self.eta * cumulative_losses)

        center_proxy = self._reputations @ stacked_states

        return local_state * (1 - self.alpha) + center_proxy * self.alpha
