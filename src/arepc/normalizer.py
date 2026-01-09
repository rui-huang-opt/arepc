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

    The paper introducing sparsemax is:
    Martins, A. F., & Astudillo, R. F. (2016).
    "From Softmax to Sparsemax: A Sparse Model of Attention and Multi-Label Classification".
    In Proceedings of the 33rd International Conference on Machine Learning (ICML).
    """

    logits_shifted: NDArray[np.float64] = logits - logits.max()
    logits_sorted = np.sort(logits_shifted)[::-1]  # Sort in descending order
    cum_sums: NDArray[np.float64] = logits_sorted.cumsum()

    condition = (1 + logits_sorted * np.arange(1, len(logits) + 1)) > cum_sums
    k = np.where(condition)[0][-1] + 1  # +1 for 1-based index

    tau: float = (cum_sums[k - 1] - 1) / k

    return np.maximum(logits_shifted - tau, 0)


def entmax15(logits: NDArray[np.float64]) -> NDArray[np.float64]:
    """
    Compute entmax with alpha=1.5 probabilities from logits.
    The formula used is:
        1.5-entmax(x) = max(0, (x / 2) - tau)^(2)
    where tau is chosen such that the output sums to 1.

    The paper introducing entmax is:
    Peters, M. E., Niculae, V., & Martins, A. F. (2019).
    "Sparse Sequence-to-Sequence Models".
    In Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics.
    """

    logits_shifted_scaled: NDArray[np.float64] = (logits - logits.max()) / 2
    logits_sorted = np.sort(logits_shifted_scaled)[::-1]
    cum_sums: NDArray[np.float64] = logits_sorted.cumsum()
    cum_squares: NDArray[np.float64] = (logits_sorted**2).cumsum()

    rho = np.arange(1, len(logits) + 1)
    m = cum_sums / rho
    s = cum_squares - rho * m**2
    delta = (1 - s) / rho
    # For numerical stability in entmax threshold computation:
    # (1) following the paper, tau candidates whose discriminant is negative would be set to +inf;
    #     here we instead mask them out using a valid mask
    # (2) the paper introduces an extra logit z_{K+1} = -inf to handle the last case;
    #     we implement this implicitly by marking the final condition as always true
    valid = delta >= 0
    tau_candidates = m - np.sqrt(np.maximum(delta, 0))
    condition = (
        (np.r_[logits_sorted[1:] <= tau_candidates[:-1], True])
        & (logits_sorted >= tau_candidates)
        & valid
    )
    k = np.where(condition)[0][0] + 1

    tau = tau_candidates[k - 1]

    return np.maximum(logits_shifted_scaled - tau, 0) ** 2


NORMALIZER_MAP = {
    "softmax": softmax,
    "sparsemax": sparsemax,
    "1.5-entmax": entmax15,
}
