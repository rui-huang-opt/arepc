from abc import ABCMeta, abstractmethod
from numpy import float64
from numpy.typing import NDArray


class DecisionRule(metaclass=ABCMeta):
    REGISTERED_SUBCLASSES: dict[str, type["DecisionRule"]] = {}

    def __init_subclass__(cls, key: str, **kwargs) -> None:
        super().__init_subclass__(**kwargs)
        cls.REGISTERED_SUBCLASSES[key] = cls

    def __init__(self) -> None:
        super().__init__()

    @classmethod
    def create(cls, key: str = "weighted_average", *args, **kwargs) -> "DecisionRule":
        if key not in cls.REGISTERED_SUBCLASSES:
            raise ValueError(f"Unknown DecisionRule key: {key}")
        return cls.REGISTERED_SUBCLASSES[key](*args, **kwargs)

    @abstractmethod
    def decide(
        self, probs: NDArray[float64], outcomes: NDArray[float64]
    ) -> NDArray[float64]: ...


class WeightedAverage(DecisionRule, key="weighted_average"):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__()

    def decide(
        self, probs: NDArray[float64], outcomes: NDArray[float64]
    ) -> NDArray[float64]:
        return probs @ outcomes
