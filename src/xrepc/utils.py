import numpy as np
from numpy.typing import NDArray
from numpy.linalg import norm


def geometric_median(
    points: NDArray[np.float64], tol: float = 1e-6, max_iter: int = 1000
) -> NDArray[np.float64]:
    """
    Compute the geometric median of a set of points using Weiszfeld's algorithm.

    Parameters:
    points (NDArray[float64]):
        An array of shape (n_points, n_dimensions) representing the points.

    tol (float):
        The tolerance for convergence. Default is 1e-5.

    max_iter (int):
        The maximum number of iterations. Default is 500.

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

    return guess
