from typing import Sequence, Protocol

import numpy as np
from numpy.typing import NDArray

from .loss_func import LOSS_FUNC_MAP
from .cumulator import CUMULATOR_MAP
from .normalizer import NORMALIZER_MAP


class LossFunc(Protocol):
    def __call__(self, neighbor_states: NDArray[np.float64]) -> NDArray[np.float64]: ...


class XRepC:
    """
    Expressive Reputation-based Consensus (XRepC) robust aggregation algorithm.

    Parameters
    ----------
    neighbors : Sequence[str]
        List of neighbor names.

    alpha : float
        Mixing parameter between local state and aggregated prediction.
        This parameter depends on the network topology and should be chosen accordingly.

    eta : float
        Temperature parameter controlling sensitivity to losses.

    loss_func : str | LossFunc, optional
        Loss function to evaluate neighbor predictions.

        If a string is provided, it should be one of the predefined loss functions.
        Options are
        "gmed" (geometric median loss),
        "qmed" (quasi-geometric median loss),
        "cmed" (coordinate-wise median loss),
        and "mean" (mean loss).
        Defaults to "cmed".

        Else, a custom loss function adhering to the LossFunc protocol:

            def custom_loss_func(neighbor_states: NDArray[np.float64]) -> NDArray[np.float64]:
                ...

        can be provided.
        This custom loss function should take in a stack of neighbor states and return a 1D array of losses.

        Note:
        In 'xrepc.loss_func.py', there is also a 'trimmed_mean_loss' function generator that can be used to create a trimmed mean loss function.
        This loss function is not directly included in the predefined loss functions as it requires the parameter `f` (number of tolerated faulty nodes), which is unknown in practice.

        To test it, you can create a loss function as follows:

            from xrepc.loss_func import trimmed_mean_loss
            custom_loss = trimmed_mean_loss(f=1)  # Example with f=1

        and then pass `custom_loss` to the `loss_func` parameter.

    cumulator : str, optional
        Cumulator type for loss aggregation.
        Options are "exp-decay" (exponentially decayed) and "moving-horizon".
        Defaults to "exp-decay".

    normalizer : str, optional
        Normalizer function to convert losses to probabilities.
        Options are "softmax", "sparsemax", and "1.5-entmax".
        Defaults to "softmax".

    horizon : float, optional
        Horizon parameter for the cumulator.

        For "moving_horizon", it defines the window size by 'horizon = int(horizon)'.

        For "forgetting_factor", it controls the decay rate by 'decay = 1 - 1 / horizon'.

        Defaults to 5.0.

    Attributes
    ----------
    probs : dict[str, float]
        Current probabilities assigned to each neighbor.

    Methods
    -------
    aggregate(local_state: NDArray[float64], neighbor_states: dict[str, NDArray[float64]]) -> NDArray[float64]
        Aggregates neighbor states with the local state using the XRepC algorithm and returns the new local state.
    """

    def __init__(
        self,
        neighbors: Sequence[str],
        alpha: float,
        eta: float,
        loss_func: str | LossFunc = "cmed",
        cumulator: str = "exp-decay",
        normalizer: str = "softmax",
        horizon: float = 5.0,
    ) -> None:
        super().__init__()

        self._neighbors = neighbors
        self._alpha = alpha
        self._eta = eta

        if isinstance(loss_func, str):
            self._loss_func = LOSS_FUNC_MAP[loss_func]
        else:
            self._loss_func = loss_func

        self._cumulator = CUMULATOR_MAP[cumulator](horizon, len(neighbors))
        self._normalizer = NORMALIZER_MAP[normalizer]

        self._probs = np.zeros(len(neighbors), dtype=np.float64)

    @property
    def probs(self) -> dict[str, float]:
        return {j: self._probs[i] for i, j in enumerate(self._neighbors)}

    def aggregate(
        self,
        local_state: NDArray[np.float64],
        neighbor_states: dict[str, NDArray[np.float64]],
    ) -> NDArray[np.float64]:
        n_states = np.stack([neighbor_states[j] for j in self._neighbors])

        current_losses = self._loss_func(n_states)
        cumulated_losses = self._cumulator(current_losses)
        self._probs = self._normalizer(-self._eta * cumulated_losses)

        honest_avarage = self._probs @ n_states

        return local_state * (1 - self._alpha) + honest_avarage * self._alpha
