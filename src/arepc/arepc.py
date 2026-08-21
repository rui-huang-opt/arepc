from typing import Literal

import numpy as np
from numpy.typing import NDArray

from .network import Network
from .loss_func import LossFunc, LOSS_FUNC_MAP
from .accumulator import make_accumulator
from .normalizer import NORMALIZER_MAP


class ARepC:
    """
    Adaptive Reputation-based Consensus (ARepC) robust aggregation algorithm.

    Parameters
    ----------
    ops : NetworkOps
        Network operations handler implementing the 'NetworkOps' protocol (defined in the 'network' module).

    alpha : float
        Mixing parameter between local state and aggregated prediction.
        This parameter depends on the network topology and should be chosen accordingly.

    eta : float
        Temperature parameter controlling sensitivity to losses.

    loss_func : LossFunc | Literal["cmed", "qmed", "gmed", "mean"], optional
        Loss function to evaluate neighbor predictions.

        If a string is provided, it selects one of the predefined loss functions.
        Options are
        "cmed" (coordinate-wise median loss),
        "qmed" (quasi-geometric median loss),
        "gmed" (geometric median loss),
        and "mean" (mean loss).
        Defaults to "cmed".

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
        from arepc import ARepC, GeometricMedianLoss, TrimmedMeanLoss

        loss_func = GeometricMedianLoss(tol=1e-5, max_iter=2000)
        # or
        loss_func = TrimmedMeanLoss(f=2)

        arepc = ARepC(neighbors, 0.5, 1.0, loss_func=loss_func)
        ```

        Besides, you can also define your own custom loss function by implementing a callable that takes
        a 2D numpy array of stacked neighbor states and returns a 1D numpy array of corresponding losses.
        e.g.,
        ```python
        from arepc import ARepC
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

        arepc = ARepC(neighbors, 0.5, 1.0, loss_func=CustomLoss(...))
        # or
        def custom_loss(neighbor_states: NDArray[np.float64]) -> NDArray[np.float64]:
            # Implement your custom loss computation here
            losses = ...
            return losses

        arepc = ARepC(neighbors, 0.5, 1.0, loss_func=custom_loss)
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
    """

    def __init__(
        self,
        network: Network,
        alpha: float,
        eta: float,
        loss_func: LossFunc | Literal["cmed", "qmed", "gmed", "mean"] = "cmed",
        accumulation: Literal["exp-decay", "moving-horizon"] = "exp-decay",
        normalization: Literal["softmax", "sparsemax", "1.5-entmax"] = "softmax",
        horizon: float = 5.0,
    ) -> None:
        self._network = network
        self.alpha = alpha
        self.eta = eta

        if isinstance(loss_func, str):
            self._loss_func = LOSS_FUNC_MAP[loss_func]
        else:
            self._loss_func = loss_func

        degree = self._network.degree

        self._accumulator = make_accumulator(accumulation, horizon, dim=degree)
        self._normalizer = NORMALIZER_MAP[normalization]

        self._reputations = np.zeros(degree, dtype=np.float64)

    @property
    def reputations(self) -> dict[str, float]:
        return {
            j: float(self._reputations[i])
            for i, j in enumerate(self._network.neighbors)
        }

    def weighted_mix(self, local_state: NDArray[np.float64]) -> NDArray[np.float64]:
        """
        The operator that aggregates neighbor states with local state using ARepC.
        The method named 'weighted_mix' is used to align with the naming convention:
            x_i(t+1) = w_ii * x_i(t) + sum_{j in N_i} w_ij * x_j(t),
        where W is the weight matrix derived from reputations.
        The difference is that here we use dynamic weights based on reputations computed from losses.
        This also facilitates easier integration with our distributed optimization framework:
            https://github.com/rui-huang-opt/discoopt.

        Args:
            local_state (NDArray[np.float64]): The local state array.

        Returns:
            NDArray[np.float64]: The updated local state array.
        """
        neighbor_states = self._network.exchange_as_array(local_state)

        current_losses = self._loss_func(neighbor_states)
        cumulative_losses = self._accumulator(current_losses)
        self._reputations = self._normalizer(-self.eta * cumulative_losses)

        center_proxy = self._reputations @ neighbor_states

        return local_state * (1 - self.alpha) + center_proxy * self.alpha
