from abc import ABCMeta, abstractmethod
from numpy import float64, mean, newaxis
from numpy.typing import NDArray
from numpy.linalg import norm


class LossFunc(metaclass=ABCMeta):
    REGISTERED_SUBCLASSES: dict[str, type["LossFunc"]] = {}

    def __init_subclass__(cls, key: str, **kwargs) -> None:
        super().__init_subclass__(**kwargs)
        cls.REGISTERED_SUBCLASSES[key] = cls

    def __init__(self) -> None:
        super().__init__()

    @classmethod
    def create(cls, key: str, *args, **kwargs) -> "LossFunc":
        if key not in cls.REGISTERED_SUBCLASSES:
            raise ValueError(f"Unknown LossFunc key: {key}")
        return cls.REGISTERED_SUBCLASSES[key](*args, **kwargs)

    def _normalize(self, losses: NDArray[float64]) -> NDArray[float64]:
        norm_losses = norm(losses)
        normalized_losses = losses / norm_losses if norm_losses > 1.0 else losses
        return normalized_losses

    @abstractmethod
    def _compute(self, outcomes: NDArray[float64]) -> NDArray[float64]: ...

    def __call__(self, outcomes: NDArray[float64]) -> NDArray[float64]:
        losses = self._compute(outcomes)
        normalized_losses = self._normalize(losses)
        return normalized_losses


class MeanDistance(LossFunc, key="mean_distance"):
    """
    Mean distance loss function implementation.
    The mean distance loss measures how far each outcome is from the others on average.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__()

    def _compute(self, outcomes: NDArray[float64]) -> NDArray[float64]:
        differences = outcomes[:, newaxis, :] - outcomes[newaxis, :, :]
        distances = norm(differences, axis=2)
        return mean(distances, axis=1)


from numpy import median


class DistanceFromMedian(LossFunc, key="distance_from_median"):
    """
    Distance from element-wise median loss function implementation.
    This loss measures how far each outcome is from the median outcome.
    The distance is computed using the L1 norm.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__()

    def _compute(self, outcomes: NDArray[float64]) -> NDArray[float64]:
        median_outcome = median(outcomes, axis=0)
        distances = norm(outcomes - median_outcome, axis=1, ord=1)
        return distances
