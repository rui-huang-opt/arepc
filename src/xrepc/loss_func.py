import numpy as np
from numpy.typing import NDArray
from numpy.linalg import norm

from .utils import geometric_median


def geometric_median_loss(neighbor_states: NDArray[np.float64]) -> NDArray[np.float64]:
    """
    Geometric median loss function implementation.
    This loss measures the Euclidean distance of each outcome from the geometric median of all outcomes.
    The drawback of this loss is its computational complexity, as finding the geometric median can be more intensive than other statistics.
    """
    outcome = geometric_median(neighbor_states)
    losses = norm(neighbor_states - outcome, axis=1)
    return losses


def quasi_geometric_median_loss(
    neighbor_states: NDArray[np.float64],
) -> NDArray[np.float64]:
    """
    Quasi-geometric median loss function implementation.
    The quasi-geometric median loss measures how far each outcome is from the others on average.
    This is an approximation of the geometric median loss that is computationally more efficient.
    """
    differences = neighbor_states[:, None, :] - neighbor_states[None, :, :]
    distances = norm(differences, axis=2)
    losses = np.mean(distances, axis=1)
    return losses


def coordinate_wise_median_loss(
    neighbor_states: NDArray[np.float64],
) -> NDArray[np.float64]:
    """
    Coordinate-wise median loss function implementation.
    This loss measures the Manhattan distance of each outcome from the coordinate-wise median of all outcomes.
    """
    outcome = np.median(neighbor_states, axis=0)
    losses = norm(neighbor_states - outcome, axis=1, ord=1)
    return losses


def mean_loss(neighbor_states: NDArray[np.float64]) -> NDArray[np.float64]:
    """
    Mean loss function implementation.
    This loss measures the Euclidean distance of each outcome from the mean of all outcomes.
    This loss is a special one as it is not supposed to be robust to outliers but empirically works well in some scenarios.
    """
    outcome = np.mean(neighbor_states, axis=0)
    losses = norm(neighbor_states - outcome, axis=1)
    return losses


LOSS_FUNC_MAP = {
    "gmed": geometric_median_loss,
    "qmed": quasi_geometric_median_loss,
    "cmed": coordinate_wise_median_loss,
    "mean": mean_loss,
}

from typing import Callable


def trimmed_mean_loss(f: int) -> Callable[[NDArray[np.float64]], NDArray[np.float64]]:
    """
    Create a trimmed mean loss function.

    This loss measures the Euclidean distance of each outcome from the trimmed mean of all outcomes.
    The trimmed mean is computed by removing the highest and lowest `f` values for each coordinate.

    The drawback of this loss is that it requires the parameter `f`, which represents the number of tolerated faulty nodes.
    However, this parameter is unknown in practice.
    Thus, this loss function is not directly included in the LOSS_FUNC_MAP.

    Parameters
    ----------
    f : int
        The number of highest and lowest values to trim for each coordinate.

    Returns
    -------
    loss_func : function
        A function that computes the trimmed mean loss.
    """

    def loss_func(neighbor_states: NDArray[np.float64]) -> NDArray[np.float64]:
        n = neighbor_states.shape[0]
        sorted_neighbor_states = neighbor_states[np.argsort(neighbor_states, axis=0)]
        trimmed_neighbor_states = sorted_neighbor_states[f : n - f]

        outcome = np.mean(trimmed_neighbor_states, axis=0)
        losses = norm(neighbor_states - outcome, axis=1)
        return losses

    return loss_func
