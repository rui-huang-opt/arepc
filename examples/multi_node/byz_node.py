import argparse


class Args(argparse.Namespace):
    idx: str
    attack: str
    n_state: int
    n_iter: int
    alpha: float


parser = argparse.ArgumentParser(description="A simple consensus test using TopoLink.")
parser.add_argument("idx", type=str, help="Index of the node.")
parser.add_argument("--attack", type=str, default="constant", help="Byzantine attack.")
parser.add_argument("--n_state", type=int, default=20, help="Dimension of the state.")
parser.add_argument("--n_iter", type=int, default=200, help="Number of iterations.")
parser.add_argument("--alpha", type=float, default=0.3, help="Step size alpha.")

args = parser.parse_args(namespace=Args())

lower_bound = -100.0
upper_bound = 100.0

import numpy.random as npr
from topolink import NodeHandle

nh = NodeHandle(args.idx)

npr.seed(int(args.idx))  # Ensure reproducibility for each node

for _ in range(args.n_iter - 1):
    attack_value = npr.uniform(-100, 100, args.n_state)
    _ = nh.laplacian(attack_value)
