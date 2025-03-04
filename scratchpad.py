import numpy as np

q = [1, 4, 2]

B = np.eye(5)

print(B[:, sorted(q)])