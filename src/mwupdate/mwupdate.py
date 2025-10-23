from numpy import float64, stack, zeros
from numpy.typing import NDArray
from .learning_rule import LearningRule
from .loss_func import LossFunc
from .decision_rule import DecisionRule


class MWUpdate:
    REGISTERED_SUBCLASSES: dict[str, type["MWUpdate"]] = {}

    def __init_subclass__(cls, key: str, **kwargs) -> None:
        super().__init_subclass__(**kwargs)
        cls.REGISTERED_SUBCLASSES[key] = cls

    def __init__(
        self,
        experts: list[str],
        learning_rule: str = "adahedge",
        loss_func: str = "pairwise_distance",
        decision_rule: str = "weighted_average",
    ) -> None:
        super().__init__()
        self._experts = experts
        self._learning_rule = LearningRule.create(self.n_experts, key=learning_rule)
        self._loss_func = LossFunc.create(key=loss_func)
        self._decision_rule = DecisionRule.create(key=decision_rule)

        self._probs: NDArray[float64] = zeros(self.n_experts)

    @property
    def n_experts(self) -> int:
        return len(self._experts)

    @property
    def probs(self) -> dict[str, float]:
        return {j: self._probs[i] for i, j in enumerate(self._experts)}

    def _compute_losses(self, outcomes: NDArray[float64]) -> NDArray[float64]:
        losses = self._loss_func(outcomes)
        max_loss = max(losses.max(), 1.0)
        normalized_losses = losses / max_loss
        return normalized_losses

    def make_decision(self, outcomes: dict[str, NDArray[float64]]) -> NDArray[float64]:
        outcomes_ = stack([outcomes[j] for j in self._experts], dtype=float64)

        losses = self._compute_losses(outcomes_)
        self._probs = self._learning_rule.compute_probs(losses)

        return self._decision_rule.decide(self._probs, outcomes_)
