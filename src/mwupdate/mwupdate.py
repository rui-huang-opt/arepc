from numpy import float64, stack, zeros
from numpy.typing import NDArray
from ._policy import Policy
from ._loss_func import LossFunc


class MWUpdate:
    REGISTERED_SUBCLASSES: dict[str, type["MWUpdate"]] = {}

    def __init_subclass__(cls, key: str, **kwargs) -> None:
        super().__init_subclass__(**kwargs)
        cls.REGISTERED_SUBCLASSES[key] = cls

    def __init__(
        self,
        experts: list[str],
        learning_rule: str = "exponentiated_gradient",
        loss_func: str = "pairwise_dispersion",
        eta: float | None = None,
    ) -> None:
        super().__init__()
        self._experts = experts
        self._policy = Policy.create(self.n_experts, eta, key=learning_rule)
        self._loss_func = LossFunc.create(key=loss_func)

        self._probs: NDArray[float64] = zeros(self.n_experts)

    @property
    def n_experts(self) -> int:
        return len(self._experts)

    @property
    def probs(self) -> dict[str, float]:
        return {j: self._probs[i] for i, j in enumerate(self._experts)}

    def make_decision(self, outcomes: dict[str, NDArray[float64]]) -> NDArray[float64]:
        outcomes_ = stack([outcomes[j] for j in self._experts], dtype=float64)

        losses = self._loss_func(outcomes_)
        self._probs = self._policy(losses)

        return self._probs @ outcomes_
