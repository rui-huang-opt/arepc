from typing import Type, TypeVar, Dict, Any, Generic

T = TypeVar("T")


class Registry(Generic[T]):
    def __init__(self):
        self._registry: Dict[str, Type[T]] = {}

    def register(self, cls: Type[T], key: str | None):
        if key is None:
            return
        if key in self._registry:
            raise ValueError(f"Class with name '{key}' is already registered.")
        self._registry[key] = cls

    def get_class(self, key: str) -> Type[T]:
        if key not in self._registry:
            raise KeyError(f"Class '{key}' not found")
        return self._registry[key]

    def create(self, key: str, *args: Any, **kwargs: Any) -> T:
        cls = self.get_class(key)
        return cls(*args, **kwargs)


from typing import Callable
from numpy import float64
from numpy.typing import NDArray

PenaltyFunc = Callable[[str, dict[str, NDArray[float64]]], float]

from abc import ABCMeta, abstractmethod
from typing import Type


class MultiplicativeWeights(metaclass=ABCMeta):
    _registry = Registry["MultiplicativeWeights"]()

    def __init_subclass__(cls, key: str | None, **kwargs) -> None:
        super().__init_subclass__(**kwargs)
        cls._registry.register(cls, key)

    def __init__(
        self, experts: list[str], penalty_func: PenaltyFunc, eps: float
    ) -> None:
        self.experts = experts
        self.eps = eps
        self.weights = {j: 1.0 for j in experts}
        self.penalty_func = penalty_func

    @classmethod
    def create(cls, key: str, *args, **kwargs) -> "MultiplicativeWeights":
        return cls._registry.create(key, *args, **kwargs)

    @property
    def normalized_weights(self) -> dict[str, float]:
        total_weight = sum(self.weights.values())
        return {j: w / total_weight for j, w in self.weights.items()}

    def update(self, outcomes: dict[str, NDArray[float64]]) -> None:
        for j in self.experts:
            penalty = self.penalty_func(j, outcomes)
            factor = 1 - self.eps * penalty
            if factor < 0:
                self.eps = 1 / penalty * 0.99
                factor = 1 - self.eps * penalty
            self.weights[j] *= factor

    @abstractmethod
    def make_prediction(
        self, outcomes: dict[str, NDArray[float64]]
    ) -> NDArray[float64]: ...


from numpy import ndarray
from numpy.random import choice


class ProbabilisticMW(MultiplicativeWeights, key="probabilistic"):
    def choose_expert(self) -> str:
        normalized_weights = self.normalized_weights
        probs = [normalized_weights[j] for j in self.experts]
        return choice(self.experts, p=probs)

    def make_prediction(
        self, outcomes: dict[str, NDArray[float64]]
    ) -> NDArray[float64]:
        expert_choice = self.choose_expert()
        return outcomes[expert_choice]


class WeightedAvgMW(MultiplicativeWeights, key="weighted_avg"):
    def combine_outcomes(
        self, outcomes: dict[str, NDArray[float64]]
    ) -> NDArray[float64]:
        normalized_weights = self.normalized_weights
        weighted_avg = sum([normalized_weights[j] * outcomes[j] for j in self.experts])

        if not isinstance(weighted_avg, ndarray):
            raise ValueError("The combined outcome is not a numpy array.")

        return weighted_avg

    def make_prediction(
        self, outcomes: dict[str, NDArray[float64]]
    ) -> NDArray[float64]:
        return self.combine_outcomes(outcomes)
