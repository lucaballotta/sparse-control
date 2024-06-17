from numpy.linalg import matrix_rank

from control_design.control_design import Designer
from control_design.cost_function import CostFunction

# import model matrices A and B
from examples.ex3 import *

# sparsity constraint
sparsity = max(len(A) - matrix_rank(A), 1)

# time horizon
h = len(A)

# cost function
cost_func = CostFunction(h, 'tr-inv')

# fully actuated
cost_fully_actuated = cost_func.compute(A, B)
print(f'cost fully actuated: {cost_fully_actuated} \n')

# s-sparse greedy
algo = 'greedy-f'
designer = Designer(A, B, sparsity, cost_func, algo)
schedule, cost = designer.design()

print('back s-sparse greedy:')
print('input schedule:', schedule)
print(f'cost: {cost} \n')

# s-sparse greedy + MCMC
designer.set_algo('mcmc')
schedule, cost = designer.design(schedule=schedule)

print('s-sparse greedy + MCMC:')
print('input schedule:', schedule)
print(f'cost: {cost} \n')

# MCMC
schedule, cost = designer.design(eps=1e-10, check_rank=True)

print('MCMC:')
print('input schedule:', schedule)
print(f'cost: {cost} \n')

# naive greedy
designer.set_algo('greedy')
schedule, cost = designer.design(eps=1e-10)

print('greedy:')
print('input schedule:', schedule)
print(f'cost: {cost}')
