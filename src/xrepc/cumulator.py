from typing import Callable

import numpy as np
from numpy.typing import NDArray


def exponentially_decayed(
    horizon: float, dim: int
) -> Callable[[NDArray[np.float64]], NDArray[np.float64]]:
    """
    Create an exponentially decayed cumulator function.

    Parameters
    ----------
    horizon : float
        The horizon parameter controlling the decay rate.
        The decay rate is computed as decay = 1 - 1 / horizon.

    dim : int
        The dimension of the input values.

    Returns
    -------
    cumulator : function
        A function that takes a new value and returns the exponentially decayed sum of all past values.
    """

    if horizon < 1.0:
        raise ValueError("Horizon must be at least 1.0.")

    decay = 1.0 - 1.0 / horizon
    cumulated_value = np.zeros(dim)

    def cumulator(new_value: NDArray[np.float64]) -> NDArray[np.float64]:
        nonlocal cumulated_value
        cumulated_value = decay * cumulated_value + new_value
        return cumulated_value

    return cumulator


def moving_horizon(
    horizon: float, dim: int
) -> Callable[[NDArray[np.float64]], NDArray[np.float64]]:
    """
    Create a moving horizon cumulator function.

    Here, we not only maintain a buffer of the most recent `horizon` values,
    but also keep track of the cumulated sum for online updates.

    Parameters
    ----------
    horizon : float
        The size of the moving horizon.
        Here, horizon will be converted to int.
        We define it as float for consistency with discounted sum cumulator.

    dim : int
        The dimension of the input values.

    Returns
    -------
    cumulator : function
        A function that takes a new value and returns the cumulated value over the moving horizon.
    """

    if horizon < 1.0:
        raise ValueError("Horizon must be at least 1.0.")

    horizon_ = int(horizon)
    buffer = np.zeros((horizon_, dim))
    index = 0
    cumulated_value = np.zeros(dim)

    def cumulator(new_value: NDArray[np.float64]) -> NDArray[np.float64]:
        nonlocal index, cumulated_value
        old_value = np.copy(buffer[index])
        buffer[index] = new_value
        cumulated_value += new_value - old_value
        index = (index + 1) % horizon_
        return cumulated_value

    return cumulator


CUMULATOR_MAP = {"exp-decay": exponentially_decayed, "moving-horizon": moving_horizon}
