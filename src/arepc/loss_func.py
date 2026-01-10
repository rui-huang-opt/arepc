import logging
from typing import Protocol

import numpy as np
from numpy.typing import NDArray
from numpy.linalg import norm
from scipy.spatial.distance import cdist

from .geometry import geometric_median, trimmed_mean

logger = logging.getLogger(__name__)


class LossFunc(Protocol):
    """
    Protocol for loss function implementations.

    Parameters
    ----------
    neighbor_states : NDArray[np.float64]
        An array of shape (n_neighbors, n_dimensions) representing the states of neighboring agents.

    Returns
    -------
    losses : NDArray[np.float64]
        An array of shape (n_neighbors,) representing the loss for each neighbor.
    """

    def __call__(self, neighbor_states: NDArray[np.float64]) -> NDArray[np.float64]: ...


def coordinate_wise_median_loss(
    neighbor_states: NDArray[np.float64],
) -> NDArray[np.float64]:
    """
    Coordinate-wise median loss function implementation.
    This loss measures the Manhattan distance of each outcome from the coordinate-wise median of all outcomes.
    """
    outcome = np.median(neighbor_states, axis=0)
    losses = norm(neighbor_states - outcome, axis=1, ord=np.inf)
    return losses


def quasi_geometric_median_loss(
    neighbor_states: NDArray[np.float64],
) -> NDArray[np.float64]:
    """
    Quasi-geometric median loss function implementation.
    The quasi-geometric median loss measures how far each outcome is from the others on average.
    This is an approximation of the geometric median loss that is computationally more efficient.
    """
    pairwise_distances = cdist(neighbor_states, neighbor_states, metric="euclidean")
    losses = np.mean(pairwise_distances, axis=1)
    return losses


class GeometricMedianLoss:
    """
    Geometric median loss function implementation.
    This loss measures the Euclidean distance of each outcome from the geometric median of all outcomes.
    The drawback of this loss is its computational complexity, as finding the geometric median can be more intensive than other statistics.

    Parameters
    ----------
    tol : float
        The tolerance for convergence in the geometric median calculation. Default is 1e-6.

    max_iter : int
        The maximum number of iterations for the geometric median calculation. Default is 1000.
    """

    def __init__(self, tol: float = 1e-6, max_iter: int = 1000) -> None:
        self.tol = tol
        self.max_iter = max_iter

    def __call__(self, neighbor_states: NDArray[np.float64]) -> NDArray[np.float64]:
        outcome = geometric_median(neighbor_states, self.tol, self.max_iter)
        losses = norm(neighbor_states - outcome, axis=1)
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


class TrimmedMeanLoss:
    """
    Trimmed mean loss function implementation.
    This loss measures the Euclidean distance of each outcome from the trimmed mean of all outcomes.
    The trimmed mean is computed by removing a certain ratio of the highest and lowest values along each dimension before calculating the mean.
    This loss provides a balance between robustness to outliers and computational efficiency.

    Parameters
    ----------
    trim_ratio : float
        The proportion of values to trim from each end of the data before computing the mean.
    """

    def __init__(self, f: int) -> None:
        if f < 0:
            err_msg = "f must be a non-negative integer."
            logger.error(err_msg)
            raise ValueError(err_msg)

        self._f = f

    def __call__(self, neighbor_states: NDArray[np.float64]) -> NDArray[np.float64]:
        outcome = trimmed_mean(neighbor_states, self._f)
        losses = norm(neighbor_states - outcome, axis=1)
        return losses


LOSS_FUNC_MAP: dict[str, LossFunc] = {
    "cmed": coordinate_wise_median_loss,
    "qmed": quasi_geometric_median_loss,
    "gmed": GeometricMedianLoss(),
    "mean": mean_loss,
}
