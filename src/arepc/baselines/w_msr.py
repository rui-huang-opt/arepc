import logging
from typing import Collection

import numpy as np
from numpy.typing import NDArray

from ..geometry import trimmed_mean

logger = logging.getLogger(__name__)


class WMSR:
    """
    Weighted Mean Subsequence Reduced (W-MSR) baseline implementation.
    This baseline computes the next state by taking a weighted average of the neighboring states,
    excluding the highest and lowest `f` values to mitigate the influence of outliers.

    Note: Standard W-MSR can only be applied to scalar states.
    For multi-dimensional states, this implementation applies W-MSR independently to each dimension.
    But this does not guarantee convergence to the true mean in multi-dimensional spaces.
    We use this approach only for simplicity and as a baseline for comparison.

    Parameters
    ----------
    f : int
        The number of highest and lowest values to exclude from the averaging process.

    Returns
    -------
    next_state : NDArray[np.float64]
        An array representing the computed next state.
    """

    def __init__(self, neighbors: Collection[str], alpha: float, f: int) -> None:
        self._neighbors = tuple(sorted(neighbors))
        self.alpha = alpha

        if f < 0:
            err_msg = "f must be a non-negative integer."
            logger.error(err_msg)
            raise ValueError(err_msg)

        self._f = f

    def mix(
        self,
        local_state: NDArray[np.float64],
        neighbor_states: dict[str, NDArray[np.float64]],
    ) -> NDArray[np.float64]:
        """
        The operator that aggregates neighbor states with local state using W-MSR.

        Args:
            local_state (NDArray[np.float64]): The local state array.

        Returns:
            NDArray[np.float64]: The updated local state array.
        """
        stacked_states = np.stack([neighbor_states[j] for j in self._neighbors], axis=0)
        center_proxy = trimmed_mean(stacked_states, self._f)
        return local_state * (1 - self.alpha) + center_proxy * self.alpha
