from numpy import float64
from numpy.typing import NDArray
from numpy.linalg import norm


def geometric_median(
    points: NDArray[float64], tol: float = 1e-5, max_iter: int = 500
) -> NDArray[float64]:
    """
    Compute the geometric median of a set of points using Weiszfeld's algorithm.

    Parameters:
    points (NDArray[float64]): An array of points with shape (n_points, n_dimensions).
    tol (float): Tolerance for convergence. The algorithm stops when the change in the median is less than this value.
    max_iter (int): Maximum number of iterations.

    Returns:
    NDArray[float64]: The geometric median of the points.
    """

    initial_guess = points.mean(axis=0)
    guess = initial_guess

    for _ in range(max_iter):
        distances = norm(points - guess, axis=1)
        nonzero_distances = distances != 0

        if not nonzero_distances.any():
            return guess

        inv_distances = 1 / distances[nonzero_distances]
        weights = inv_distances / inv_distances.sum()
        new_guess = (weights[:, None] * points[nonzero_distances]).sum(axis=0)

        if ((new_guess - guess) ** 2).sum() < tol**2:
            return new_guess

        guess = new_guess

    return guess
