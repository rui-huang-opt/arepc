import logging

import numpy as np
from numpy.typing import NDArray

logger = logging.getLogger(__name__)


def trimmed_mean(data: NDArray[np.float64], n_trim: int = 1) -> NDArray[np.float64]:
    """
    Compute the coordinate-wise trimmed mean of the data by removing the n_trim smallest and largest values.
    The data is assumed to be a 2D array where each row is a sample and each column is a feature.
    When 2 * n_trim >= n_samples, it reduces n_trim to ensure at least one value remains.
    When n_trim <= 0, it returns the regular mean.

    Parameters:
        data (NDArray[float64]):
            An array of shape (n_samples, n_features).

        n_trim (int):
            The number of smallest and largest values to trim from each feature.

    Returns:
        NDArray[float64]: The trimmed mean of the data.
    """
    if 2 * n_trim >= data.shape[0]:
        n_trim = data.shape[0] // 2 - 1

    if n_trim <= 0:
        return data.mean(axis=0)

    sorted_data = np.sort(data, axis=0)
    trimmed_data = sorted_data[n_trim:-n_trim]
    return trimmed_data.mean(axis=0)
