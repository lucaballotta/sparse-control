import numpy as np
import networkx as nx

n = 50
m = 50
d = .1

G = nx.random_geometric_graph(n, d)
A = nx.to_numpy_array(G) / n
B = np.eye(m)