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


from abc import ABCMeta, abstractmethod
from typing import Type
from numpy import float64
from numpy.typing import NDArray


class MultiplicativeWeights(metaclass=ABCMeta):
    _registry = Registry["MultiplicativeWeights"]()

    def __init_subclass__(cls, key: str | None, **kwargs) -> None:
        super().__init_subclass__(**kwargs)
        cls._registry.register(cls, key)

    def __init__(self, experts: list[str], eps: float) -> None:
        self._experts = experts
        self._eps = eps
        self._weights = {j: 1.0 for j in experts}

    @classmethod
    def create(cls, key: str, *args, **kwargs) -> "MultiplicativeWeights":
        return cls._registry.create(key, *args, **kwargs)

    @property
    def normalized_weights(self) -> dict[str, float]:
        total_weight = sum(self._weights.values())
        return {j: w / total_weight for j, w in self._weights.items()}

    def update(self, outcomes: dict[str, NDArray[float64]]) -> None:
        for j in self._experts:
            penalty = penalty_func(j, outcomes)
            factor = 1 - self._eps * penalty
            if factor < 0:
                self._eps = 1 / penalty * 0.99
                factor = 1 - self._eps * penalty
            self._weights[j] *= factor

    @abstractmethod
    def make_prediction(
        self, outcomes: dict[str, NDArray[float64]]
    ) -> NDArray[float64]: ...


from numpy import ndarray
from numpy.random import choice


class ProbabilisticMW(MultiplicativeWeights, key="probabilistic"):
    def _choose_expert(self) -> str:
        normalized_weights = self.normalized_weights
        probs = [normalized_weights[j] for j in self._experts]
        return choice(self._experts, p=probs)

    def make_prediction(
        self, outcomes: dict[str, NDArray[float64]]
    ) -> NDArray[float64]:
        expert_choice = self._choose_expert()
        return outcomes[expert_choice]


class WeightedAvgMW(MultiplicativeWeights, key="weighted_avg"):
    def _combine_outcomes(
        self, outcomes: dict[str, NDArray[float64]]
    ) -> NDArray[float64]:
        normalized_weights = self.normalized_weights
        weighted_avg = sum([normalized_weights[j] * outcomes[j] for j in self._experts])

        if not isinstance(weighted_avg, ndarray):
            raise ValueError("The combined outcome is not a numpy array.")

        return weighted_avg

    def make_prediction(
        self, outcomes: dict[str, NDArray[float64]]
    ) -> NDArray[float64]:
        return self._combine_outcomes(outcomes)


from numpy import mean
from numpy.linalg import norm


def penalty_func(j: str, x_js: dict[str, NDArray[float64]]) -> float:
    x_j = x_js[j]
    return mean([norm(x_j - x_js[k]) for k in x_js], dtype=float64)
