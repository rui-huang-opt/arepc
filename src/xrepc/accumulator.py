from typing import Callable, Literal

import numpy as np
from numpy.typing import NDArray


def make_accumulator(
    accumulation: Literal["exp-decay", "moving-horizon"], horizon: float, dim: int
) -> Callable[[NDArray[np.float64]], NDArray[np.float64]]:
    """
    Factory function to create an accumulator function based on the specified method.

    Parameters
    ----------
    accumulation : Literal["exp-decay", "moving-horizon"]
        The accumulation method to use.
        Options are "exp-decay (exponentially decayed)" and "moving-horizon".

    horizon : float
        The horizon parameter for the accumulation method.

        For "moving_horizon", it defines the window size by 'horizon = int(horizon)'.

        For "forgetting_factor", it controls the decay rate by 'decay = 1 - 1 / horizon'.
        This is because the effective memory length of an exponentially decayed accumulator is approximately '1 / (1 - decay)'.

    dim : int
        The dimension of the input values.

    Returns
    -------
    accumulator : Callable[[NDArray[np.float64]], NDArray[np.float64]]
        A function that takes a new value and returns the accumulated value based on the specified method.
    """

    if horizon < 1.0:
        raise ValueError("Horizon must be at least 1.0.")

    cumulative_value = np.zeros(dim, dtype=np.float64)

    if accumulation == "exp-decay":
        decay = 1.0 - 1.0 / horizon

        def accumulator(new_value: NDArray[np.float64]) -> NDArray[np.float64]:
            nonlocal cumulative_value
            cumulative_value = decay * cumulative_value + new_value
            return cumulative_value

        return accumulator

    elif accumulation == "moving-horizon":
        horizon_ = int(horizon)
        buffer = np.zeros((horizon_, dim), dtype=np.float64)
        index = 0

        def accumulator(new_value: NDArray[np.float64]) -> NDArray[np.float64]:
            nonlocal index, cumulative_value
            old_value = np.copy(buffer[index, :])
            buffer[index, :] = new_value
            cumulative_value += new_value - old_value
            index = (index + 1) % horizon_
            return cumulative_value

        return accumulator

    else:
        raise ValueError(f"Unknown accumulation method: {accumulation}")
