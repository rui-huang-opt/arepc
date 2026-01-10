import math
import logging
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class TverbergNode:
    point: NDArray[np.float64]
    proof: list[list[int]]


def radon(
    points: NDArray[np.float64],
) -> tuple[NDArray[np.float64], list[list[int]]]:
    """
    Compute the Radon point of a set of points in R^d.

    Parameters:
        points (NDArray[float64]):
            An array of shape (d + 2, d) representing the points.

    Returns:
        tuple[NDArray[float64], list[list[int]]]:
            A tuple containing the Radon point and the proof (partition of points).
    """
    aug_points = np.hstack((points, np.ones((points.shape[0], 1))))
    _, _, vh = np.linalg.svd(aug_points.T)
    null_space_vector = vh[-1, :]
    u1 = np.where(null_space_vector > 0)[0].tolist()
    u2 = np.where(null_space_vector < 0)[0].tolist()

    weight = null_space_vector[u1]
    radon_point = (weight @ points[u1] / weight.sum())[:-1]

    return radon_point, [u1, u2]


def prune(
    point: NDArray[np.float64], indices: list[int], data: NDArray[np.float64]
) -> list[int]:
    n_features = data.shape[1]

    if len(indices) <= n_features + 1:
        return indices

    while len(indices) > n_features + 1:
        selected_points = data[indices, :]
        n_indices = len(indices)

        aug_points = np.hstack((selected_points, np.ones((n_indices, 1))))

        _, _, vh = np.linalg.svd(aug_points.T)
        null_space_vector = vh[-1, :]

        to_remove = np.argmax(np.abs(null_space_vector)).item()
        indices.pop(to_remove)

    return indices


def iterated_tverberg(data: NDArray[np.float64]) -> TverbergNode:
    """
    Compute the Tverberg point of the data using an iterative approximation method.

    Parameters:
        data (NDArray[float64]):
            An array of shape (n_samples, n_features).

    Returns:
        NDArray[float64]: The approximated Tverberg point of the data.
    """
    n_samples, n_features = data.shape
    n_stacks = math.ceil(math.log2(n_samples / (2 * (n_features + 1) ** 2)))
    stacks: list[list[TverbergNode]] = [[] for _ in range(n_stacks + 1)]

    for i in range(n_samples):
        t_node = TverbergNode(point=data[i], proof=[[i]])
        stacks[0].append(t_node)

    while not stacks[n_stacks]:
        new_proof: list[list[int]] = []

        for ell in range(n_stacks, 0, -1):
            if len(stacks[ell - 1]) >= n_features + 2:
                break

        qs = [stacks[ell - 1].pop() for _ in range(n_features + 2)]
        aug_points = np.array([q.point for q in qs])
        radon_point, radon_proof = radon(aug_points)

        depth_of_qj = 2 ** (ell - 1)

        for k in range(2):
            for i in range(depth_of_qj):
                x: list[int] = []
                for j in radon_proof[k]:
                    x.extend(qs[j].proof[i])

                x_pruned = prune(radon_point, x, data)
                new_proof.append(x_pruned)

                x_recycled = [j for j in x if j not in x_pruned]

                for j in x_recycled:
                    t_node = TverbergNode(point=data[j], proof=[[j]])
                    stacks[0].append(t_node)

        t_node = TverbergNode(point=radon_point, proof=new_proof)
        stacks[ell].append(t_node)

    return stacks[n_stacks][0]


# This function is not tested yet.
# Don't use it in production.
