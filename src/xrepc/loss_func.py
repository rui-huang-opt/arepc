import numpy as np
from numpy.typing import NDArray
from numpy.linalg import norm

from .utils import geometric_median


def geometric_median_loss(advices: NDArray[np.float64]) -> NDArray[np.float64]:
    """
    Geometric median loss function implementation.
    This loss measures the Euclidean distance of each outcome from the geometric median of all outcomes.
    The drawback of this loss is its computational complexity, as finding the geometric median can be more intensive than other statistics.
    """
    outcome = geometric_median(advices)
    losses = norm(advices - outcome, axis=1)
    return losses


def quasi_geometric_median_loss(advices: NDArray[np.float64]) -> NDArray[np.float64]:
    """
    Quasi-geometric median loss function implementation.
    The quasi-geometric median loss measures how far each outcome is from the others on average.
    This is an approximation of the geometric median loss that is computationally more efficient.
    """
    differences = advices[:, None, :] - advices[None, :, :]
    distances = norm(differences, axis=2)
    losses = np.mean(distances, axis=1)
    return losses


def coordinate_wise_median_loss(advices: NDArray[np.float64]) -> NDArray[np.float64]:
    """
    Coordinate-wise median loss function implementation.
    This loss measures the Manhattan distance of each outcome from the coordinate-wise median of all outcomes.
    """
    outcome = np.median(advices, axis=0)
    losses = norm(advices - outcome, axis=1, ord=1)
    return losses


def mean_loss(advices: NDArray[np.float64]) -> NDArray[np.float64]:
    """
    Mean loss function implementation.
    This loss measures the Euclidean distance of each outcome from the mean of all outcomes.
    This loss is a special one as it is not supposed to be robust to outliers but empirically works well in some scenarios.
    """
    outcome = np.mean(advices, axis=0)
    losses = norm(advices - outcome, axis=1)
    return losses


LOSS_FUNC_MAP = {
    "gm": geometric_median_loss,
    "qgm": quasi_geometric_median_loss,
    "cwm": coordinate_wise_median_loss,
    "mean": mean_loss,
}


class TrimmedMeanLoss:
    """
    Trimmed mean loss function implementation.
    This loss measures the Euclidean distance of each outcome from the trimmed mean of all outcomes.
    The trimmed mean is computed by removing the highest and lowest 10% of values for each coordinate as default.
    The drawback of this loss is that it requires parameter tuning (the trimming fraction) to achieve optimal performance,
    and thus is not practical in real-world scenarios.
    This is just a demonstration of how to implement such a loss function.
    """

    def __init__(self, f: int) -> None:
        self._f = f

    def __call__(self, advices: NDArray[np.float64]) -> NDArray[np.float64]:
        n = advices.shape[0]
        sorted_advices = advices[np.argsort(advices, axis=0)]
        trimmed_advices = sorted_advices[self._f : n - self._f]

        outcome = np.mean(trimmed_advices, axis=0)
        losses = norm(advices - outcome, axis=1)
        return losses
