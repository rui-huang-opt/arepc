import argparse


class Args(argparse.Namespace):
    idx: str
    n_state: int
    n_iter: int
    alpha: float


parser = argparse.ArgumentParser(description="A simple consensus test using TopoLink.")
parser.add_argument("idx", type=str, help="Index of the node.")
parser.add_argument("--n_state", type=int, default=20, help="Dimension of the state.")
parser.add_argument("--n_iter", type=int, default=200, help="Number of iterations.")
parser.add_argument("--alpha", type=float, default=0.3, help="Step size alpha.")

args = parser.parse_args(namespace=Args())

import numpy as np
from numpy.random import uniform, seed
from topolink import NodeHandle

lower_bound = -100.0
upper_bound = 100.0

nh = NodeHandle(args.idx)

from arepc import ARepC

agent = ARepC(nh, args.alpha, eta=0.02, loss_func="qmed", normalization="1.5-entmax")
x = np.zeros((args.n_iter, args.n_state))
seed(int(args.idx))  # Ensure reproducibility for each node
x[0] = uniform(lower_bound, upper_bound, args.n_state)

for k in range(args.n_iter - 1):
    # Compute new state
    x[k + 1] = agent.weighted_mix(x[k])

    print(f"Node {args.idx} at iteration {k + 1}: state: {x[k + 1]}")
    print(f"Node {args.idx} at iteration {k + 1}: reputations: {agent.reputations}")
