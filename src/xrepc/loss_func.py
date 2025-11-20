from numpy import float64, mean
from numpy.typing import NDArray
from numpy.linalg import norm


def quasi_geometric_median_loss(advices: NDArray[float64]) -> NDArray[float64]:
    """
    Quasi-geometric median loss function implementation.
    The quasi-geometric median loss measures how far each outcome is from the others on average.
    """
    differences = advices[:, None, :] - advices[None, :, :]
    distances = norm(differences, axis=2)
    losses = mean(distances, axis=1)
    return losses


from numpy import median


def coordinate_wise_median_loss(advices: NDArray[float64]) -> NDArray[float64]:
    """
    Coordinate-wise median loss function implementation.
    This loss measures the Manhattan distance of each outcome from the coordinate-wise median of all outcomes.
    """
    median_advice = median(advices, axis=0)
    losses = norm(advices - median_advice, axis=1, ord=1)
    return losses


from .utils import geometric_median


def geometric_median_loss(advices: NDArray[float64]) -> NDArray[float64]:
    """
    Geometric median loss function implementation.
    This loss measures the Euclidean distance of each outcome from the geometric median of all outcomes.
    """
    geometric_median_advice = geometric_median(advices)
    losses = norm(advices - geometric_median_advice, axis=1)
    return losses


def mean_loss(advices: NDArray[float64]) -> NDArray[float64]:
    """
    Mean loss function implementation.
    This loss measures the Euclidean distance of each outcome from the mean of all outcomes.
    """
    mean_advice = mean(advices, axis=0)
    losses = norm(advices - mean_advice, axis=1)
    return losses


from numpy import argsort


class TrimmedMeanLoss:
    """
    Trimmed mean loss function implementation.
    This loss measures the Euclidean distance of each outcome from the trimmed mean of all outcomes.
    The trimmed mean is computed by removing the highest and lowest 10% of values for each coordinate.
    """

    def __init__(self, trim_fraction: float = 0.1) -> None:
        self._trim_fraction = trim_fraction

    def __call__(self, advices: NDArray[float64]) -> NDArray[float64]:
        n = advices.shape[0]
        k = int(n * self._trim_fraction)

        sorted_advices = advices[argsort(advices, axis=0)]
        trimmed_advices = sorted_advices[k : n - k]

        trimmed_mean_advice = mean(trimmed_advices, axis=0)
        losses = norm(advices - trimmed_mean_advice, axis=1)
        return losses
