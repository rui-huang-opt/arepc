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


from numpy.random import choice


class ProbabilisticSelection(DecisionRule, key="probabilistic_selection"):
    def __init__(self) -> None:
        super().__init__()

    def decide(
        self, probs: NDArray[float64], outcomes: NDArray[float64]
    ) -> NDArray[float64]:
        chosen_index = choice(len(outcomes), p=probs)
        return outcomes[chosen_index]


class WeightedAverage(DecisionRule, key="weighted_average"):
    def __init__(self) -> None:
        super().__init__()

    def decide(
        self, probs: NDArray[float64], outcomes: NDArray[float64]
    ) -> NDArray[float64]:
        return probs @ outcomes


from numpy import where, average
from .baselines.repc import min_f


class ProbabilityTrimmed(DecisionRule, key="probability_trimmed"):
    def __init__(self, f: int, eps: float = 0.001) -> None:
        super().__init__()
        self._f = f
        self._eps = eps
        self._eps_t = eps

    def decide(
        self, probs: NDArray[float64], outcomes: NDArray[float64]
    ) -> NDArray[float64]:
        min_f_prob = min_f(probs, f=self._f)
        if min_f_prob is None:
            return average(outcomes, axis=0)

        weights = where(probs > min_f_prob, 1.0, self._eps_t)

        self._eps_t *= self._eps

        return average(outcomes, axis=0, weights=weights)


from numpy import clip, inf


class BoundedWeightedAverage(DecisionRule, key="bounded_weighted_average"):
    def __init__(self, min_outcome: float = -inf, max_outcome: float = inf) -> None:
        super().__init__()
        self._min_outcome = min_outcome
        self._max_outcome = max_outcome

    def decide(
        self, probs: NDArray[float64], outcomes: NDArray[float64]
    ) -> NDArray[float64]:
        outcomes = clip(outcomes, self._min_outcome, self._max_outcome)

        return probs @ outcomes
