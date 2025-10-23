from typing import Protocol
from numpy import float64
from numpy.typing import NDArray


class LossFunc(Protocol):
    def __call__(self, outcomes: NDArray[float64]) -> NDArray[float64]:
        """Computes the loss for each expert given their outcomes."""
        ...


class DecisionRule(Protocol):
    def decide(
        self, probs: NDArray[float64], outcomes: NDArray[float64]
    ) -> NDArray[float64]:
        """Makes a decision based on the experts' probabilities and outcomes."""
        ...


from abc import ABCMeta, abstractmethod


class MWUpdate(metaclass=ABCMeta):
    REGISTERED_SUBCLASSES: dict[str, type["MWUpdate"]] = {}

    def __init_subclass__(cls, key: str, **kwargs) -> None:
        super().__init_subclass__(**kwargs)
        cls.REGISTERED_SUBCLASSES[key] = cls

    def __init__(
        self, experts: list[str], loss_func: LossFunc, decision_rule: DecisionRule
    ) -> None:
        super().__init__()
        self._experts = experts
        self._loss_func = loss_func
        self._decision_rule = decision_rule

        self._probs: NDArray[float64] = ones(len(experts)) / self.n_experts
        self._cumulative_losses: NDArray[float64] = zeros(self.n_experts)

    @property
    def n_experts(self) -> int:
        return len(self._experts)

    @property
    def probs(self) -> dict[str, float]:
        return {j: self._probs[i] for i, j in enumerate(self._experts)}

    @classmethod
    def create(
        cls,
        experts: list[str],
        loss_func: LossFunc,
        decision_rule: DecisionRule,
        key: str = "adahedge",
        *args,
        **kwargs,
    ) -> "MWUpdate":
        if key not in cls.REGISTERED_SUBCLASSES:
            raise ValueError(f"Unknown MWUpdate key: {key}")
        subclass = cls.REGISTERED_SUBCLASSES[key]
        return subclass(experts, loss_func, decision_rule, *args, **kwargs)

    @abstractmethod
    def make_decision(self, outcomes: dict[str, NDArray[float64]]) -> NDArray[float64]:
        pass


from numpy import stack
from numpy import ones, zeros, exp, sqrt, log


class Hedge(MWUpdate, key="hedge"):
    def __init__(
        self,
        experts: list[str],
        loss_func: LossFunc,
        decision_rule: DecisionRule,
        eta: float | None = None,
    ) -> None:
        super().__init__(experts, loss_func, decision_rule)

        self._eta = eta
        self._t = 1

    @property
    def probs(self) -> dict[str, float]:
        return {j: self._probs[i] for i, j in enumerate(self._experts)}

    @property
    def n_experts(self) -> int:
        return len(self._experts)

    @property
    def eta(self) -> float:
        if self._eta is None:
            return sqrt(log(self.n_experts) / self._t)
        else:
            return self._eta

    def _compute_losses(self, outcomes: NDArray[float64]) -> NDArray[float64]:
        losses = self._loss_func(outcomes)
        max_loss = max(losses.max(), 1.0)
        normalized_losses = losses / max_loss
        return normalized_losses

    def _update_probs(self, losses: NDArray[float64]) -> None:
        self._cumulative_losses += losses
        weights = exp(-self.eta * self._cumulative_losses)
        self._probs = weights / weights.sum()

    def make_decision(self, outcomes: dict[str, NDArray[float64]]) -> NDArray[float64]:
        outcomes_ = stack([outcomes[j] for j in self._experts], dtype=float64)

        losses = self._compute_losses(outcomes_)
        self._update_probs(losses)

        self._t += 1

        return self._decision_rule.decide(self._probs, outcomes_)


from numpy import inf
from numpy import where


class AdaHedge(MWUpdate, key="adahedge"):
    def __init__(
        self, experts: list[str], loss_func: LossFunc, decision_rule: DecisionRule
    ) -> None:
        super().__init__(experts, loss_func, decision_rule)

        self._cumulative_mix_gap = 0.0

    def _compute_eta(self) -> float:
        if self._cumulative_mix_gap == 0.0:
            return inf
        else:
            return log(self.n_experts) / self._cumulative_mix_gap

    def _compute_losses(self, outcomes: NDArray[float64]) -> NDArray[float64]:
        losses = self._loss_func(outcomes)
        max_loss = max(losses.max(), 1.0)
        normalized_losses = losses / max_loss
        return normalized_losses

    def _mix(self, eta: float) -> tuple[NDArray[float64], float]:
        min_cum_loss: float = self._cumulative_losses.min()

        if eta == inf:
            weights = where(self._cumulative_losses == min_cum_loss, 1.0, 0.0)
        else:
            weights = exp(-eta * (self._cumulative_losses - min_cum_loss))

        weights_sum = weights.sum()

        probs = weights / weights_sum
        mix_loss = min_cum_loss - log(weights_sum / self.n_experts) / eta

        return probs, mix_loss

    def _update_probs(self, losses: NDArray[float64]) -> None:
        eta = self._compute_eta()

        self._probs, mix_loss_prev = self._mix(eta)
        mixed_loss = self._probs @ losses
        self._cumulative_losses += losses

        _, mix_loss = self._mix(eta)

        mix_gap = max(0.0, mixed_loss - mix_loss + mix_loss_prev)
        self._cumulative_mix_gap += mix_gap

    def make_decision(self, outcomes: dict[str, NDArray[float64]]) -> NDArray[float64]:
        outcomes_ = stack([outcomes[j] for j in self._experts], axis=0)

        losses = self._compute_losses(outcomes_)
        self._update_probs(losses)

        return self._decision_rule.decide(self._probs, outcomes_)
