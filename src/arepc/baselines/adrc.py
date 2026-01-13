import logging
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from ..network import NetworkOps
from ..geometry import iterated_radon, iterated_tverberg

logger = logging.getLogger(__name__)


class ADRC:
    """
    Approximate Distributed Robust Convergence (ADRC) baseline implementation.
    Here, we use the improved version of ADRC that uses approximated centerpoints computed via
    iterated Radon (or iterated Tverberg) algorithms.

    Note:
    For 1D data, the geometric median can be computed exactly in linear time.
    For 2D and 3D data, there exist more efficient and exact algorithms to compute centerpoints.
    However, this implementation uses iterated Radon/Tverberg for benchmarking with our proposed methods, which work in arbitrary dimensions.

    Parameters
    ----------
    ops : NetworkOps
        The network operations interface.

    alpha : float
        The mixing parameter for the weighted average.

    method : Literal["radon", "tverberg"], optional
        The method to use for centerpoint estimation.
        "radon" for Iterated Radon method (default).
        "tverberg" for Iterated Tverberg method.
        The latter has better theoretical guarantees but is more computationally expensive.
        Thus, Iterated Radon is often preferred in practice, though it only offers probabilistic guarantees.

    Returns
    -------
    next_state : NDArray[np.float64]
        An array representing the computed next state.
    """

    def __init__(
        self,
        ops: NetworkOps,
        alpha: float,
        method: Literal["radon", "tverberg"] = "radon",
    ) -> None:
        self._ops = ops
        self.alpha = alpha

        if method == "radon":
            self._c_estimator = iterated_radon
        elif method == "tverberg":

            def estimator(data: NDArray[np.float64]) -> NDArray[np.float64]:
                cp = iterated_tverberg(data)
                return cp.point

            self._c_estimator = estimator
        else:
            err_msg = f"Unknown method '{method}'. Supported methods are 'radon' and 'tverberg'."
            logger.error(err_msg)
            raise ValueError(err_msg)

    def weighted_mix(self, local_state: NDArray[np.float64]) -> NDArray[np.float64]:
        """
        The operator that aggregates neighbor states with local state using ADRC.

        Args:
            local_state (NDArray[np.float64]): The local state array.

        Returns:
            NDArray[np.float64]: The updated local state array.
        """
        neighbor_states = self._ops.exchange_as_array(local_state)
        center_proxy = self._c_estimator(neighbor_states)

        return local_state * (1 - self.alpha) + center_proxy * self.alpha
