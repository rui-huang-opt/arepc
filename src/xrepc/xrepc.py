from numpy import float64, stack, zeros
from numpy.typing import NDArray
from .loss_func import LossFunc
from .utils import softmax


class XRepC:
    def __init__(
        self,
        neighbors: list[str],
        alpha: float,
        eta: float,
        loss_func: str = "quasi_geometric_median",
        decay: float = 0.3,
    ) -> None:
        super().__init__()
        self._neighbors = neighbors
        self._alpha = alpha
        self._eta = eta
        self._loss_func = LossFunc.create(key=loss_func)

        if not (0.0 <= decay <= 1.0):
            raise ValueError("decay factor must be in [0, 1].")

        self._decay = decay

        self._losses_memory = zeros(self.n_neighbors, dtype=float64)
        self._probs = zeros(self.n_neighbors, dtype=float64)

    @property
    def n_neighbors(self) -> int:
        return len(self._neighbors)

    @property
    def probs(self) -> dict[str, float]:
        return {j: self._probs[i] for i, j in enumerate(self._neighbors)}

    def _update_probs(self, outcomes: NDArray[float64]) -> None:
        losses = self._loss_func(outcomes)

        self._losses_memory *= self._decay
        self._losses_memory += losses

        self._probs = softmax(-self._eta * self._losses_memory)

    def aggregate(
        self,
        local_state: NDArray[float64],
        neighbor_states: dict[str, NDArray[float64]],
    ) -> NDArray[float64]:
        outcomes = stack([neighbor_states[j] for j in self._neighbors])
        self._update_probs(outcomes)
        prediction = self._probs @ outcomes

        return (1 - self._alpha) * local_state + self._alpha * prediction
