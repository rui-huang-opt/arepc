from logging import basicConfig, INFO

basicConfig(level=INFO)

from topolink import Graph

NORMAL_NODES = ["1", "2", "3", "4", "5"]
NORMAL_EDGES = [("1", "2"), ("2", "3"), ("3", "4"), ("4", "5"), ("5", "1"), ("1", "3")]
BYZANTINE_NODES = ["6", "7", "8", "9", "10"]
BYZANTINE_EDGES = [("1", "6"), ("2", "7"), ("3", "8"), ("4", "9"), ("1", "10")]

NODES = NORMAL_NODES + BYZANTINE_NODES
EDGES = NORMAL_EDGES + BYZANTINE_EDGES

N_NORMAL = len(NORMAL_NODES)
N_BYZANTINE = len(BYZANTINE_NODES)
N_NODES = N_NORMAL + N_BYZANTINE

graph = Graph(NODES, EDGES)

graph.deploy()
