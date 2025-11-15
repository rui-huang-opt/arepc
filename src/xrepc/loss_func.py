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
    def create(cls, *args, key: str, **kwargs) -> "LossFunc":
        if key not in cls.REGISTERED_SUBCLASSES:
            raise ValueError(f"Unknown LossFunc key: {key}")
        return cls.REGISTERED_SUBCLASSES[key](*args, **kwargs)

    @abstractmethod
    def __call__(self, outcomes: NDArray[float64]) -> NDArray[float64]: ...


class QuasiGeometricMedian(LossFunc, key="quasi_geometric_median"):
    """
    Quasi-geometric median loss function implementation.
    The quasi-geometric median loss measures how far each outcome is from the others on average.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__()

    def __call__(self, outcomes: NDArray[float64]) -> NDArray[float64]:
        differences = outcomes[:, newaxis, :] - outcomes[newaxis, :, :]
        distances = norm(differences, axis=2)
        losses = mean(distances, axis=1)
        return losses


from numpy import median


class CoordinateWiseMedian(LossFunc, key="coordinate_wise_median"):
    """
    Distance from coordinate-wise median loss function implementation.
    This loss measures the L1 distance of each outcome from the coordinate-wise median of all outcomes.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__()

    def __call__(self, outcomes: NDArray[float64]) -> NDArray[float64]:
        median_outcome = median(outcomes, axis=0)
        losses = norm(outcomes - median_outcome, axis=1)
        return losses


from .utils import geometric_median


class GeometricMedian(LossFunc, key="geometric_median"):
    """
    Geometric median loss function implementation.
    This loss measures the Euclidean distance of each outcome from the geometric median of all outcomes.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__()

    def __call__(self, outcomes: NDArray[float64]) -> NDArray[float64]:
        geo_median = geometric_median(outcomes)
        losses = norm(outcomes - geo_median, axis=1)
        return losses
