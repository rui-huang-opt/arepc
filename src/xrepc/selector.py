import numpy as np
from numpy.typing import NDArray


def softmax(logits: NDArray[np.float64]) -> NDArray[np.float64]:
    """
    Compute softmax probabilities from logits.
    The formula used is:
        softmax(x_i) = exp(x_i - max(x)) / sum_j exp(x_j - max(x))
    This formulation improves numerical stability by subtracting the maximum logit.
    """

    logits_shifted: NDArray[np.float64] = logits - logits.max()
    weights: NDArray[np.float64] = np.exp(logits_shifted)

    return weights / weights.sum()


def sparsemax(logits: NDArray[np.float64]) -> NDArray[np.float64]:
    """
    Compute sparsemax probabilities from logits.
    The formula used is:
        sparsemax(x) = max(0, x - tau)
    where tau is chosen such that the output sums to 1.
    """

    logits_shifted: NDArray[np.float64] = logits - logits.max()
    logits_sorted = np.sort(logits_shifted)[::-1]  # Sort in descending order
    cum_sums: NDArray[np.float64] = logits_sorted.cumsum()
    mask = (1 + logits_sorted * np.arange(1, len(logits) + 1)) > cum_sums
    k = np.count_nonzero(mask)

    tau: float = (cum_sums[k - 1] - 1) / k
    probs = logits_shifted - tau
    probs[probs < 0] = 0.0

    return probs


def entmax15(logits: NDArray[np.float64]) -> NDArray[np.float64]:
    """
    Compute entmax with alpha=1.5 probabilities from logits.
    The formula used is:
        entmax15(x) = max(0, (x - tau) / 2)^(2)
    where tau is chosen such that the output sums to 1.
    """
    logits_shifted: NDArray[np.float64] = (logits - logits.max()) / 2
    logits_sorted = np.sort(logits_shifted)[::-1]
    cum_sums: NDArray[np.float64] = logits_sorted.cumsum()
    cum_squares: NDArray[np.float64] = (logits_sorted**2).cumsum()
    rho = np.arange(1, len(logits) + 1)

    m = cum_sums / rho
    s = cum_squares - rho * m**2
    delta = np.maximum((1 - s) / rho, 0)  # For numerical stability
    tau_candidates = m - np.sqrt(delta)

    for k in range(len(logits) - 1):
        if logits_sorted[k + 1] <= tau_candidates[k] <= logits_sorted[k]:
            tau = tau_candidates[k]
            break
    else:
        tau = tau_candidates[-1]

    probs = np.maximum(0, logits_shifted - tau) ** 2

    return probs


SELECTOR_MAP = {
    "softmax": softmax,
    "sparsemax": sparsemax,
    "entmax15": entmax15,
}
