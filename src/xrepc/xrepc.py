from numpy import float64, stack, empty, exp
from numpy.typing import NDArray
from ._loss_func import LossFunc


class XRepC:
    def __init__(
        self, neighbors: list[str], eta: float, loss_func: str = "pairwise_dispersion"
    ) -> None:
        super().__init__()
        self._neighbors = neighbors
        self._eta = eta
        self._loss_func = LossFunc.create(key=loss_func)
        self._weights = empty(self.n_neighbors, dtype=float64)

    @property
    def n_neighbors(self) -> int:
        return len(self._neighbors)

    @property
    def weights(self) -> dict[str, float]:
        return {j: self._weights[i] for i, j in enumerate(self._neighbors)}

    def _update_weights(self, losses: NDArray[float64]) -> None:
        weights = exp(-self._eta * losses)
        self._weights = weights / weights.sum()

    def make_decision(
        self, neighbor_states: dict[str, NDArray[float64]]
    ) -> NDArray[float64]:
        n_states = stack([neighbor_states[j] for j in self._neighbors], dtype=float64)

        losses = self._loss_func(n_states)
        self._update_weights(losses)

        return self._weights @ n_states
