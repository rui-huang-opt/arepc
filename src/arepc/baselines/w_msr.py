import logging

import numpy as np
from numpy.typing import NDArray

from ..network import NetworkOps
from ..utils import trimmed_mean


logger = logging.getLogger(__name__)


class WMSR:
    """
    Weighted Mean Subsequence Reduced (W-MSR) baseline implementation.
    This baseline computes the next state by taking a weighted average of the neighboring states,
    excluding the highest and lowest `f` values to mitigate the influence of outliers.

    Parameters
    ----------
    f : int
        The number of highest and lowest values to exclude from the averaging process.

    Returns
    -------
    next_state : NDArray[np.float64]
        An array representing the computed next state.
    """

    def __init__(self, ops: NetworkOps, alpha: float, f: int) -> None:
        self._ops = ops
        self.alpha = alpha

        if f < 0:
            err_msg = "f must be a non-negative integer."
            logger.error(err_msg)
            raise ValueError(err_msg)

        self._f = f

    def weighted_mix(self, local_state: NDArray[np.float64]) -> NDArray[np.float64]:
        """
        The operator that aggregates neighbor states with local state using W-MSR.

        Args:
            local_state (NDArray[np.float64]): The local state array.

        Returns:
            NDArray[np.float64]: The updated local state array.
        """
        neighbor_states = self._ops.exchange_as_array(local_state)
        neighbor_estimate = trimmed_mean(neighbor_states, self._f)
        return local_state * (1 - self.alpha) + neighbor_estimate * self.alpha
