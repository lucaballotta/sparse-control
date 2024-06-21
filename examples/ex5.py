import numpy as np
import networkx as nx

n = 100
m = 100
d = .1

G = nx.random_geometric_graph(n, d)
# A = np.array(nx.laplacian_matrix(G)) / n
A = nx.to_numpy_array(G) /n
B = np.eye(m)