from curses.ascii import BS
from numpy.linalg import matrix_rank

from control_design.control_design import Designer
from control_design.cost_function import CostFunction

# import model matrices A and B
from examples.ex1 import *

# sparsity constraint
sparsity = max(len(A) - matrix_rank(A), 1)

# time horizon
h = len(A)

# cost function
cost_func = CostFunction(h, 'lambda-min')

# fully actuated
cost_fully_actuated = cost_func.compute(A, B)
print(f'cost fully actuated: {cost_fully_actuated} \n')

# all actuator active at last time step
B_all_last = [np.zeros_like(B), B]
cost_all_last = cost_func.compute(A, B_all_last)
print(f'cost all actuator active at last time step: {cost_all_last} \n')

# forward greedy
algo = 'greedy-f'
designer = Designer(A, B, sparsity, cost_func, algo)
schedule, cost = designer.design()

print('forward greedy:')
print('input schedule:', schedule)
print(f'cost: {cost} \n')

# forward greedy + MCMC
designer.set_algo('mcmc')
schedule, cost = designer.design(schedule=schedule)

print('forward greedy + MCMC:')
print('input schedule:', schedule)
print(f'cost: {cost} \n')

# MCMC
schedule, cost = designer.design(eps=1e-10)

print('MCMC:')
print('input schedule:', schedule)
print(f'cost: {cost} \n')

# naive greedy
designer.set_algo('greedy')
schedule, cost = designer.design(eps=1e-10)

print('greedy:')
print('input schedule:', schedule)
print(f'cost: {cost}')
