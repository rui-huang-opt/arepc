from numpy import float64
from numpy.typing import NDArray
from abc import ABCMeta, abstractmethod


class LearningRule(metaclass=ABCMeta):
    REGISTERED_SUBCLASSES: dict[str, type["LearningRule"]] = {}

    def __init_subclass__(cls, key: str, **kwargs) -> None:
        super().__init_subclass__(**kwargs)
        cls.REGISTERED_SUBCLASSES[key] = cls

    def __init__(self, n_experts: int) -> None:
        super().__init__()
        self._n_experts = n_experts
        self._cumulative_losses: NDArray[float64] = zeros(n_experts)

    @classmethod
    def create(
        cls, n_experts: int, key: str = "adahedge", *args, **kwargs
    ) -> "LearningRule":
        if key not in cls.REGISTERED_SUBCLASSES:
            raise ValueError(f"Unknown MWUpdate key: {key}")
        subclass = cls.REGISTERED_SUBCLASSES[key]
        return subclass(n_experts, *args, **kwargs)

    @abstractmethod
    def compute_probs(self, losses: NDArray[float64]) -> NDArray[float64]:
        """Compute updated probabilities based on losses."""
        ...


from numpy import ones, zeros, exp, sqrt, log


class Hedge(LearningRule, key="hedge"):
    def __init__(self, n_experts: int, eta: float | None = None) -> None:
        super().__init__(n_experts)

        self._eta = eta
        self._t = 1

    @property
    def eta(self) -> float:
        if self._eta is None:
            return sqrt(log(self._n_experts) / self._t)
        else:
            return self._eta

    def compute_probs(self, losses: NDArray[float64]) -> NDArray[float64]:
        self._cumulative_losses += losses
        weights = exp(-self.eta * self._cumulative_losses)
        probs = weights / weights.sum()
        self._t += 1
        return probs


from numpy import inf
from numpy import where


class AdaHedge(LearningRule, key="adahedge"):
    def __init__(self, n_experts: int) -> None:
        super().__init__(n_experts)

        self._cumulative_mix_gap = 0.0

    def _compute_eta(self) -> float:
        if self._cumulative_mix_gap == 0.0:
            return inf
        else:
            return log(self._n_experts) / self._cumulative_mix_gap

    def _mix(self, eta: float) -> tuple[NDArray[float64], float]:
        min_cum_loss: float = self._cumulative_losses.min()

        if eta == inf:
            weights = where(self._cumulative_losses == min_cum_loss, 1.0, 0.0)
        else:
            weights = exp(-eta * (self._cumulative_losses - min_cum_loss))

        weights_sum = weights.sum()

        probs = weights / weights_sum
        mix_loss = min_cum_loss - log(weights_sum / self._n_experts) / eta

        return probs, mix_loss

    def compute_probs(self, losses: NDArray[float64]) -> NDArray[float64]:
        eta = self._compute_eta()

        probs, mix_loss_prev = self._mix(eta)
        mixed_loss = probs @ losses
        self._cumulative_losses += losses

        _, mix_loss = self._mix(eta)

        mix_gap = max(0.0, mixed_loss - mix_loss + mix_loss_prev)
        self._cumulative_mix_gap += mix_gap

        return probs
