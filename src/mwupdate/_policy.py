from abc import ABCMeta, abstractmethod
from numpy import float64
from numpy.typing import NDArray


class Policy(metaclass=ABCMeta):
    REGISTERED_SUBCLASSES: dict[str, type["Policy"]] = {}

    def __init_subclass__(cls, key: str, **kwargs) -> None:
        super().__init_subclass__(**kwargs)
        cls.REGISTERED_SUBCLASSES[key] = cls

    def __init__(self, n_experts: int) -> None:
        super().__init__()
        self._n_experts = n_experts

    @classmethod
    def create(cls, n_experts: int, *args, key: str = "adahedge", **kwargs) -> "Policy":
        if key not in cls.REGISTERED_SUBCLASSES:
            raise ValueError(f"Unknown MWUpdate key: {key}")
        subclass = cls.REGISTERED_SUBCLASSES[key]
        return subclass(n_experts, *args, **kwargs)

    @abstractmethod
    def __call__(self, losses: NDArray[float64]) -> NDArray[float64]:
        """Compute updated probabilities based on losses."""
        ...


from numpy import exp, zeros, roll


class ExponentiatedGradient(Policy, key="exponentiated_gradient"):
    def __init__(
        self, n_experts: int, eta: float, memory: int = 1, *args, **kwargs
    ) -> None:
        super().__init__(n_experts)

        self._eta = eta
        self._losses_history = zeros((memory, self._n_experts), dtype=float64)

    def __call__(self, losses: NDArray[float64]) -> NDArray[float64]:
        self._losses_history = roll(self._losses_history, shift=-1, axis=0)
        self._losses_history[-1, :] = losses
        weights = exp(-self._eta * self._losses_history.mean(axis=0))
        probs = weights / weights.sum()
        return probs


# TODO: Verify AdaHedge implementation to continuous version. i.e., Exponential Gradient algorithm.
