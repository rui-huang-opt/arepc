from typing import Protocol
from numpy import float64
from numpy.typing import NDArray


class LossFunc(Protocol):
    """
    Protocol for loss function implementations.
    Any loss function class should implement the __call__ method that takes
    an array of outcomes and returns an array of losses.
    """

    def __call__(self, outcomes: NDArray[float64]) -> NDArray[float64]: ...


from numpy import mean
from numpy.linalg import norm


class QuasiGeometricMedianLoss:
    """
    Quasi-geometric median loss function implementation.
    The quasi-geometric median loss measures how far each outcome is from the others on average.
    """

    def __init__(self) -> None:
        super().__init__()

    def __call__(self, outcomes: NDArray[float64]) -> NDArray[float64]:
        differences = outcomes[:, None, :] - outcomes[None, :, :]
        distances = norm(differences, axis=2)
        losses = mean(distances, axis=1)
        return losses


from numpy import median


class CoordinateWiseMedianLoss:
    """
    Distance from coordinate-wise median loss function implementation.
    This loss measures the L1 distance of each outcome from the coordinate-wise median of all outcomes.
    """

    def __init__(self) -> None:
        super().__init__()

    def __call__(self, outcomes: NDArray[float64]) -> NDArray[float64]:
        median_outcome = median(outcomes, axis=0)
        losses = norm(outcomes - median_outcome, axis=1)
        return losses


from .utils import geometric_median


class GeometricMedianLoss:
    """
    Geometric median loss function implementation.
    This loss measures the Euclidean distance of each outcome from the geometric median of all outcomes.
    """

    def __init__(self) -> None:
        super().__init__()

    def __call__(self, outcomes: NDArray[float64]) -> NDArray[float64]:
        geo_median = geometric_median(outcomes)
        losses = norm(outcomes - geo_median, axis=1)
        return losses


class MeanLoss:
    """
    Mean loss function implementation.
    This loss measures the Euclidean distance of each outcome from the mean of all outcomes.
    """

    def __init__(self) -> None:
        super().__init__()

    def __call__(self, outcomes: NDArray[float64]) -> NDArray[float64]:
        mean_outcome = mean(outcomes, axis=0)
        losses = norm(outcomes - mean_outcome, axis=1)
        return losses


from numpy import argsort


class TrimmedMeanLoss:
    """
    Trimmed mean loss function implementation.
    This loss measures the Euclidean distance of each outcome from the trimmed mean of all outcomes.
    The trimmed mean is computed by removing the highest and lowest 10% of values for each coordinate.
    """

    def __init__(self, trim_fraction: float = 0.1) -> None:
        super().__init__()

        if not (0.0 <= trim_fraction < 0.5):
            raise ValueError("trim_fraction must be in [0.0, 0.5).")

        self._trim_fraction = trim_fraction

    def __call__(self, outcomes: NDArray[float64]) -> NDArray[float64]:
        n = outcomes.shape[0]
        k = int(n * self._trim_fraction)

        sorted_outcomes = outcomes[argsort(outcomes, axis=0)]
        trimmed_outcomes = sorted_outcomes[k : n - k]

        trimmed_mean_outcome = mean(trimmed_outcomes, axis=0)
        losses = norm(outcomes - trimmed_mean_outcome, axis=1)
        return losses
