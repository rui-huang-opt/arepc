from typing import Sequence, Literal

import numpy as np
from numpy.typing import NDArray

from .loss_func import LossFunc, LOSS_FUNC_MAP
from .accumulator import make_accumulator
from .normalizer import NORMALIZER_MAP


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

    loss_func : LossFunc | Literal["qmed", "cmed", "mean"], optional
        Loss function to evaluate neighbor predictions.

        If a string is provided, it selects one of the predefined loss functions.
        Options are
        "qmed" (quasi-geometric median loss),
        "cmed" (coordinate-wise median loss),
        "gmed" (geometric median loss),
        and "mean" (mean loss).
        Defaults to "qmed".

        There are also two additional loss functions that require configuration:
        - GeometricMedianLoss:
          This loss function computes the geometric median of neighbor states iteratively using Weiszfeld's algorithm.
          Thus, it has two parameters: `tol` (tolerance for convergence) and `max_iter` (maximum number of iterations).
          The 'gmed' option uses a default instance of this class with `tol=1e-6` and `max_iter=1000`.
          You can customize these parameters by instantiating the class separately and passing the instance as `loss_func`.
        - TrimmedMeanLoss:
          This loss function requires specifying the trimming parameter `f`, which is usually unknown in practice.
          So this loss function is not recommended for general use.
          However, if you want to experiment with it, you can instantiate the class separately and pass the instance as `loss_func`.
        To use these, instantiate them separately and pass the instance as `loss_func`.
        e.g.,
        ```python
        from xrepc import XRepC, GeometricMedianLoss, TrimmedMeanLoss

        loss_func = GeometricMedianLoss(tol=1e-5, max_iter=2000)
        # or
        loss_func = TrimmedMeanLoss(f=2)

        xrepc = XRepC(neighbors, 0.5, 1.0, loss_func=loss_func)
        ```

        Besides, you can also define your own custom loss function by implementing a callable that takes
        a 2D numpy array of stacked neighbor states and returns a 1D numpy array of corresponding losses.
        e.g.,
        ```pythonpython
        from xrepc import XRepC
        from numpy.typing import NDArray
        import numpy as np

        class CustomLoss:
            def __init__(self, ...) -> None:
                # Initialize any parameters needed for your custom loss
                ...

            def __call__(self, neighbor_states: NDArray[np.float64]) -> NDArray[np.float64]:
                # Implement your custom loss computation here
                losses = ...
                return losses

        xrepc = XRepC(neighbors, 0.5, 1.0, loss_func=CustomLoss(...))
        # or
        def custom_loss(neighbor_states: NDArray[np.float64]) -> NDArray[np.float64]:
            # Implement your custom loss computation here
            losses = ...
            return losses

        xrepc = XRepC(neighbors, 0.5, 1.0, loss_func=custom_loss)
        ```

    accumulation : str, optional
        Accumulation type for loss aggregation.
        Options are "exp-decay" (exponentially decayed) and "moving-horizon".
        Defaults to "exp-decay".

    normalization : str, optional
        Normalization function to convert losses to probabilities.
        Options are "softmax", "sparsemax", and "1.5-entmax".
        Defaults to "softmax".

    horizon : float, optional
        Horizon parameter for the accumulation method.

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
        loss_func: LossFunc | Literal["qmed", "cmed", "gmed", "mean"] = "qmed",
        accumulation: Literal["exp-decay", "moving-horizon"] = "exp-decay",
        normalization: Literal["softmax", "sparsemax", "1.5-entmax"] = "softmax",
        horizon: float = 5.0,
    ) -> None:
        self._neighbors = neighbors
        self.alpha = alpha
        self.eta = eta

        if isinstance(loss_func, str):
            self._loss_func = LOSS_FUNC_MAP[loss_func]
        else:
            self._loss_func = loss_func

        self._accumulator = make_accumulator(accumulation, horizon, len(neighbors))
        self._normalizer = NORMALIZER_MAP[normalization]

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
        cumulative_losses = self._accumulator(current_losses)
        self._probs = self._normalizer(-self.eta * cumulative_losses)

        honest_avarage = self._probs @ n_states

        return local_state * (1 - self.alpha) + honest_avarage * self.alpha
