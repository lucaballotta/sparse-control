import sys

import numpy as np
import control as ct

from control_design.utils import get_q_redundant_controllable

# Set the parameters
n_x = 10
n_u = 10
n_static_total = 3  # Set to None to ignore

# Set the requirements
requirements_met = False

n_iter = 0
while not requirements_met:
    if n_static_total is not None and n_iter == 0:
        print(f"Trying to find a system with {n_static_total} static states... checked:")
    n_iter += 1
    print(f"{n_iter} systems", end='\r', flush=True)
    #: Generate a system
    sys = ct.drss(states=n_x, inputs=n_u, outputs=n_x)
    A, B = sys.A, sys.B
    #: Compute the requirements
    q_redundant = get_q_redundant_controllable(A, B)
    p_redundant = get_q_redundant_controllable(A.T, np.eye(n_x))
    #: Check the requirements
    if n_static_total is not None and q_redundant <= n_x - n_static_total:
        requirements_met = True
        print(f"Found a system with {n_static_total} static states after {n_iter + 1} iterations, sensors needed: {n_x - p_redundant}")
        #