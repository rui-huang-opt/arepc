from numpy import float64, stack, exp, zeros, roll
from numpy.typing import NDArray
from ._loss_func import LossFunc


class XRepC:
    def __init__(
        self,
        neighbors: list[str],
        alpha: float,
        eta: float,
        loss_func: str = "distance_from_median",
        decay_factor: float = 0.3,
    ) -> None:
        super().__init__()
        self._neighbors = neighbors
        self._alpha = alpha
        self._eta = eta
        self._loss_func = LossFunc.create(key=loss_func)

        if not (0.0 <= decay_factor <= 1.0):
            raise ValueError("decay_factor must be in [0, 1].")

        self._decay_factor = decay_factor

        self._losses = zeros(self.n_neighbors, dtype=float64)
        self._losses_memory = zeros(self.n_neighbors, dtype=float64)
        self._probs = zeros(self.n_neighbors, dtype=float64)

    @property
    def n_neighbors(self) -> int:
        return len(self._neighbors)

    @property
    def losses(self) -> dict[str, float]:
        return {j: self._losses[i] for i, j in enumerate(self._neighbors)}

    @property
    def probs(self) -> dict[str, float]:
        return {j: self._probs[i] for i, j in enumerate(self._neighbors)}

    def _update_probs(self, outcomes: NDArray[float64]) -> None:
        self._losses = self._loss_func(outcomes)
        self._losses_memory *= self._decay_factor
        self._losses_memory += self._losses
        weights = exp(-self._eta * self._losses_memory)
        self._probs = weights / weights.sum()

    def aggregate(
        self,
        local_state: NDArray[float64],
        neighbor_states: dict[str, NDArray[float64]],
    ) -> NDArray[float64]:
        n_states = stack([neighbor_states[j] for j in self._neighbors], dtype=float64)
        self._update_probs(n_states)

        return (1 - self._alpha) * local_state + self._alpha * (self._probs @ n_states)
