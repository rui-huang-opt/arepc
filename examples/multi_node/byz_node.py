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

import numpy as np
from numpy.random import seed, uniform
from topolink import NodeHandle

nh = NodeHandle(args.idx)

x = np.zeros((args.n_iter, args.n_state))
seed(int(args.idx))  # Ensure reproducibility for each node
x[0] = uniform(lower_bound, upper_bound, args.n_state)
for k in range(args.n_iter - 1):
    laplacian = nh.laplacian(x[k])

    if args.attack == "constant":
        x[k + 1] = x[k]
    elif args.attack == "random":
        x[k + 1] = uniform(lower_bound, upper_bound, args.n_state)
    elif args.attack == "disturbance":
        disturbance = uniform(-1.0, 1.0, args.n_state)
        x[k + 1] = x[k] - laplacian * args.alpha + disturbance
    elif args.attack == "coordination":
        disturbance = np.zeros(args.n_state)
        disturbance[0] = uniform(lower_bound, upper_bound)
        x[k + 1] = x[k] - laplacian * args.alpha + disturbance
