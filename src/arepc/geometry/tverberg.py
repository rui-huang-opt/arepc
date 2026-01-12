import math
import logging
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.linalg import null_space
from scipy.optimize import linprog

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class CertifiedPoint:
    """
    A data structure to hold a point and its certification proof.

    Attributes:
        point (NDArray[float64]):
            A point in d-dimensional space.

        proof (list[list[int]]):
            A list of lists, where each sublist contains the indices of points
            that certify the its depth.
    """

    point: NDArray[np.float64]
    proof: list[list[int]]


def affine_dependence(points: NDArray[np.float64]) -> NDArray[np.float64]:
    """
    Compute an affine dependence among the given points.

    Parameters:
        points (NDArray[float64]):
            An array of shape (n_samples, n_features).

    Returns:
        NDArray[float64]:
            A vector of coefficients representing the affine dependence.
    """
    n_samples = points.shape[0]
    aug_points = np.vstack([points.T, np.ones(n_samples)])
    ns = null_space(aug_points)

    if ns.size == 0:
        err_msg = "No affine dependence found among the given points."
        logger.error(err_msg)
        raise ValueError(err_msg)

    return ns[:, 0]


def convex_combination(
    target: NDArray[np.float64], points: NDArray[np.float64]
) -> NDArray[np.float64]:
    """
    Compute a convex combination of the given points.

    Parameters:
        points (NDArray[float64]):
            An array of shape (n_samples, n_features).

    Returns:
        NDArray[float64]:
            A vector of coefficients representing the convex combination.
    """
    n_samples = points.shape[0]
    c = np.zeros(n_samples)
    A_eq = np.vstack([points.T, np.ones(n_samples)])
    b_eq = np.r_[target, 1]
    bounds = [(0, None) for _ in range(n_samples)]

    res = linprog(c, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method="highs")

    if not res["success"]:
        err_msg = f"Linear programming failed: {res['message']}"
        logger.error(err_msg)
        raise ValueError(err_msg)

    return res["x"]


def radon(points: NDArray[np.float64]) -> CertifiedPoint:
    """
    Radon partitioning of d+2 points in d-dimensional space.

    Parameters:
        points (NDArray[float64]):
            An array of shape (d+2, d).

    Returns:
        CertifiedPoint:
            A CertifiedPoint object containing the Radon point and the proof (partition of points).
    """
    ad = affine_dependence(points)
    pos_indices = np.where(ad > 0)[0]
    neg_indices = np.where(ad < 0)[0]

    weight = ad[pos_indices]
    radon_point = weight @ points[pos_indices] / weight.sum()

    return CertifiedPoint(radon_point, [pos_indices.tolist(), neg_indices.tolist()])


def prune(
    point: NDArray[np.float64], indices: list[int], data: NDArray[np.float64]
) -> list[int]:
    n_features = data.shape[1]

    if len(indices) <= n_features + 1:
        return indices

    while len(indices) > n_features + 1:
        selected_points = data[indices, :]
        cc = convex_combination(point, selected_points)
        ad = affine_dependence(selected_points)

        pos = np.where(ad > 0)[0]

        if pos.size == 0:
            err_msg = "Affine dependence has no positive coefficients."
            logger.error(err_msg)
            raise ValueError(err_msg)

        ratios = cc[pos] / ad[pos]
        remove_index = pos[np.argmin(ratios)]
        indices.pop(remove_index)

    return indices


def iterated_tverberg(data: NDArray[np.float64]) -> CertifiedPoint:
    """
    Compute the Tverberg point of the data using an iterative approximation method.

    The paper introducing IteratedTverberg is:
    Miller, G. L., & Sheehy, D. R. (2009).
    “Approximate Center Points with Proofs.”
    In Proceedings of the Twenty-Fifth Annual Symposium on Computational Geometry, pp. 153-158.

    Parameters:
        data (NDArray[float64]):
            An array of shape (n_samples, n_features).

    Returns:
        CertifiedPoint: The approximated Tverberg point of the data.
    """
    n_samples, n_features = data.shape
    n_buckets = math.ceil(math.log2(n_samples / (2 * (n_features + 1) ** 2)))

    if n_buckets <= 0:
        err_msg = (
            "The number of samples is too small to compute a Tverberg point. "
            f"At least {2 * (n_features + 1) ** 2} samples are required. "
            f"But got {n_samples} samples."
        )
        logger.error(err_msg)
        raise ValueError(err_msg)

    buckets: list[list[CertifiedPoint]] = [[] for _ in range(n_buckets + 1)]
    for i in range(n_samples):
        c_point = CertifiedPoint(point=data[i, :], proof=[[i]])
        buckets[0].append(c_point)

    while not buckets[n_buckets]:
        new_proof: list[list[int]] = []

        for ell in range(n_buckets, 0, -1):
            if len(buckets[ell - 1]) >= n_features + 2:
                break

        qs = [buckets[ell - 1].pop() for _ in range(n_features + 2)]
        radon_point = radon(np.array([cp.point for cp in qs]))

        depth_of_qj = 2 ** (ell - 1)

        for k in range(2):
            for i in range(depth_of_qj):
                x: list[int] = []
                for j in radon_point.proof[k]:
                    x.extend(qs[j].proof[i])

                x_pruned = prune(radon_point.point, x, data)
                new_proof.append(x_pruned)

                x_recycled = [j for j in x if j not in x_pruned]

                for j in x_recycled:
                    c_point = CertifiedPoint(point=data[j], proof=[[j]])
                    buckets[0].append(c_point)

        c_point = CertifiedPoint(point=radon_point.point, proof=new_proof)
        buckets[ell].append(c_point)

    return buckets[n_buckets][0]
