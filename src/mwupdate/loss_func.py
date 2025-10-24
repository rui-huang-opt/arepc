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
    def create(cls, key: str = "pairwise_distance", *args, **kwargs) -> "LossFunc":
        if key not in cls.REGISTERED_SUBCLASSES:
            raise ValueError(f"Unknown LossFunc key: {key}")
        return cls.REGISTERED_SUBCLASSES[key](*args, **kwargs)

    @abstractmethod
    def __call__(self, outcomes: NDArray[float64]) -> NDArray[float64]: ...


class PairwiseDistanceLoss(LossFunc, key="pairwise_distance"):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__()

    def __call__(self, outcomes: NDArray[float64]) -> NDArray[float64]:
        differences = outcomes[:, newaxis, :] - outcomes[newaxis, :, :]
        distances = norm(differences, axis=2)
        return mean(distances, axis=1)


from numpy import median


class MedianDistanceLoss(LossFunc, key="median_distance"):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__()

    def __call__(self, outcomes: NDArray[float64]) -> NDArray[float64]:
        median_outcome = median(outcomes, axis=0)
        distances = norm(outcomes - median_outcome, axis=1)
        return distances
