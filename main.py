from numpy.linalg import matrix_rank

from control_design.control_design import Designer
from control_design.cost_function import CostFunction

# import model matrices A and B
from examples.ex1 import *

# sparsity constraint
s = max(len(A) - matrix_rank(A), 1)

# time horizon
h = len(A)

# cost function
cost_func = CostFunction(h, 'inv-tr')

# fully actuated
cost_fully_actuated = cost_func.compute(A, B)
print('cost fully actuated', cost_fully_actuated)

# all actuator active at last time step
B_all_last = [np.zeros_like(B), B]
cost_all_last = cost_func.compute(A, B_all_last)
print('cost all actuator active at last time step', cost_all_last)

# sparsity constraint
sparsity = 'pw'

# algorithm
algo = 'greedy-f'

# find sparsity schedule
designer = Designer(A, B, cost_func, s, sparsity, algo)
schedule, cost = designer.design()

print('greedy:')
print('input schedule:', schedule)
print('cost', cost)

designer.set_algo('mcmc')
schedule, cost = designer.design(schedule=schedule)

print('greedy + MCMC:')
print('input schedule:', schedule)
print('cost', cost)

schedule, cost = designer.design()

print('MCMC:')
print('input schedule:', schedule)
print('cost', cost)