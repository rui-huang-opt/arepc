def min_f(scores: dict[str, float], f: int) -> float | None:
    """
    The min-f score is defined as the f-th smallest unique score in the dictionary,
    but it is not allowed to be the largest score.
    If there is only one unique score, return None to indicate that all scores are equal.
    """

    if not scores:
        raise ValueError("Scores dictionary is empty.")

    deduped_scores = sorted(set(scores.values()))

    if len(deduped_scores) == 1:
        return None

    if f >= len(deduped_scores):
        return deduped_scores[-2]
    else:
        return deduped_scores[f - 1]
