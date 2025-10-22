from numpy import unique
from numpy import float64
from numpy.typing import NDArray


def min_f(probs: NDArray[float64], f: int) -> float | None:
    """
    The min-f score is defined as the f-th smallest unique score in the array,
    but it is not allowed to be the largest score.
    If there is only one unique score, return None to indicate that all scores are equal.
    """

    if probs.size == 0:
        raise ValueError("Probabilities array is empty.")

    deduped_probs = unique(probs)

    if len(deduped_probs) == 1:
        return None

    if f >= len(deduped_probs):
        return deduped_probs[-2]
    else:
        return deduped_probs[f - 1]
