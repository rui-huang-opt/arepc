from typing import Protocol, Sequence
from numpy import float64, stack, zeros
from numpy.typing import NDArray
from .loss_func import cwm_loss
from .utils import softmax, sparsemax


class LossFunc(Protocol):
    """
    Protocol for loss function implementations.
    Any loss function class should implement the __call__ method that takes
    an array of advices and returns an array of losses.
    The method does not require an explicit outcome input, as the outcome is
    derived from the advices within the loss function itself.
    """

    def __call__(self, advices: NDArray[float64]) -> NDArray[float64]: ...


class XRepC:
    """
    Exponential Replicator Dynamics for Collaborative Learning (XRepC).

    Parameters
    ----------
    neighbors : Sequence[str]
        List of neighbor identifiers.

    alpha : float
        Mixing parameter between local state and aggregated prediction.

    eta : float
        Learning rate for updating probabilities.

    loss_func : LossFunc, optional
        Loss function to evaluate neighbor advices. Defaults to coordinate-wise median loss.

    decay : float, optional
        Decay factor for past losses, must be in [0, 1]. Defaults to 0.3.

    Attributes
    ----------
    n_neighbors : int
        Number of neighbors.

    probs : dict[str, float]
        Current probabilities assigned to each neighbor.

    Methods
    -------
    aggregate(local_state: NDArray[float64], neighbor_states: dict[str, NDArray[float64]]) -> NDArray[float64]
        Aggregates neighbor states with the local state using the XRepC algorithm.
    """

    def __init__(
        self,
        neighbors: Sequence[str],
        alpha: float,
        eta: float,
        loss_func: LossFunc | None = None,
        decay: float = 0.8,
    ) -> None:
        super().__init__()

        self._neighbors = neighbors
        self._alpha = alpha
        self._eta = eta
        self._loss_func = cwm_loss if loss_func is None else loss_func

        if not (0.0 <= decay <= 1.0):
            raise ValueError("decay factor must be in [0, 1].")

        self._decay = decay

        self._total_expert_losses = zeros(self.n_neighbors, dtype=float64)
        self._probs = zeros(self.n_neighbors, dtype=float64)

    @property
    def n_neighbors(self) -> int:
        return len(self._neighbors)

    @property
    def probs(self) -> dict[str, float]:
        return {j: self._probs[i] for i, j in enumerate(self._neighbors)}

    def aggregate(
        self,
        local_state: NDArray[float64],
        neighbor_states: dict[str, NDArray[float64]],
    ) -> NDArray[float64]:
        advices = stack([neighbor_states[j] for j in self._neighbors])

        expert_losses = self._loss_func(advices)

        self._total_expert_losses *= self._decay
        self._total_expert_losses += expert_losses

        self._probs = sparsemax(-self._eta * self._total_expert_losses)

        prediction = self._probs @ advices

        return local_state * (1 - self._alpha) + prediction * self._alpha
