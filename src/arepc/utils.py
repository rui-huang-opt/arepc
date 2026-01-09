import logging

import numpy as np
from numpy.typing import NDArray
from numpy.linalg import norm

logger = logging.getLogger(__name__)


def geometric_median(
    points: NDArray[np.float64], tol: float = 1e-6, max_iter: int = 1000
) -> NDArray[np.float64]:
    """
    Compute the geometric median of a set of points using Weiszfeld's algorithm.

    Parameters:
    points (NDArray[float64]):
        An array of shape (n_points, n_dimensions) representing the points.

    tol (float):
        The tolerance for convergence. Default is 1e-6.

    max_iter (int):
        The maximum number of iterations. Default is 1000.

    Returns:
    NDArray[float64]: The geometric median of the points.
    """

    guess: NDArray[np.float64] = points.mean(axis=0)

    for _ in range(max_iter):
        distances: NDArray[np.float64] = norm(points - guess, axis=1)
        nonzero_mask: NDArray[np.bool_] = distances != 0.0

        if not nonzero_mask.any():
            return guess

        inv_distances = 1 / distances[nonzero_mask]
        weights: NDArray[np.float64] = inv_distances / inv_distances.sum()
        new_guess = weights @ points[nonzero_mask]

        if ((new_guess - guess) ** 2).sum() < tol**2:
            return new_guess

        guess = new_guess

    logger.warning(
        f"Geometric median did not converge within {max_iter} iterations."
        "The result may be inaccurate. Try increasing `max_iter` or `tol`."
    )

    return guess


def trimmed_mean(data: NDArray[np.float64], n_trim: int = 1) -> NDArray[np.float64]:
    """
    Compute the trimmed mean of the data by removing the n_trim smallest and largest values.
    The data is assumed to be a 2D array where each row is a sample and each column is a feature.

    Parameters:
        data (NDArray[float64]):
            An array of shape (n_samples, n_features).

        n_trim (int):
            The number of smallest and largest values to trim from each feature.

    Returns:
        NDArray[float64]: The trimmed mean of the data.
    """
    if n_trim <= 0:
        warn_msg = f"n_trim should be positive, got {n_trim}. Returning regular mean."
        logger.warning(warn_msg)
        return data.mean(axis=0)

    if 2 * n_trim >= data.shape[0]:
        n_trim = data.shape[0] // 2 - 1
        warn_msg = f"n_trim too large for sample size, reduced to {n_trim}."
        logger.warning(warn_msg)

    n = data.shape[0]
    sorted_data = np.sort(data, axis=0)
    trimmed_data = sorted_data[n_trim : n - n_trim]
    return trimmed_data.mean(axis=0)
