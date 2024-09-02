import datetime
import os.path
import pickle
from numpy.linalg import matrix_rank

from control_design.control_design import Designer
from control_design.cost_function import CostFunction

# to save results 
save_result = True

# import model matrices A and B
exp_id = 5
if exp_id == 1:
    from examples.ex1 import *
elif exp_id == 2:
    from examples.ex2 import *
elif exp_id == 3:
    from examples.ex3 import *
elif exp_id == 4:
    from examples.ex4 import *
elif exp_id == 5:
    from examples.ex5 import *

h = len(A)
cost = 'logdet'

# sparsity constraints
s_init = max(len(A) - matrix_rank(A), 1)
s_step = 2
s_max = min(s_init + 10, len(B[0]) - s_step)
s_vec = range(s_init, s_max, s_step)

# build output vectors
schedule_s_greedy_all = dict.fromkeys(s_vec)
cost_s_greedy_all = dict.fromkeys(s_vec)
schedule_s_greedy_mcmc_all = dict.fromkeys(s_vec)
cost_s_greedy_mcmc_all = dict.fromkeys(s_vec)

# find sparsity schedules
cost_func = CostFunction(h, cost)
designer = Designer(A, B, s_init, cost_func)

for s in s_vec:
    print(f'sparsity: {s} \n')
    designer.set_sparsity(s)

    # s-sparse greedy
    designer.set_algo('greedy-f')
    schedule_s_greedy, cost_s_greedy = designer.design()
    schedule_s_greedy = [schedule_k for schedule_k in schedule_s_greedy if len(schedule_k) > 0]
    cost_s_greedy_all[s] = cost_s_greedy
    schedule_s_greedy_all[s] = schedule_s_greedy
    print('s-sparse greedy:')
    print(f'cost: {cost_s_greedy} \n')

    # s-sparse MCMC with warm start
    designer.set_algo('mcmc')
    schedule_s_greedy_mcmc, cost_s_greedy_mcmc = designer.design(schedule=schedule_s_greedy)
    cost_s_greedy_mcmc_all[s] = cost_s_greedy_mcmc
    schedule_s_greedy_mcmc_all[s] = schedule_s_greedy_mcmc
    print('s-sparse greedy + MCMC:')
    print(f'cost: {cost_s_greedy_mcmc} \n')

if save_result:
    dir_name = 'exp'
    if not os.path.isdir(dir_name):
        os.mkdir(dir_name)

    file_name = 'ex' + str(exp_id) + '_s_' + cost + '_' + datetime.datetime.now().strftime('%Y%m%d%I%M')
    with open(dir_name + '/' + file_name + '.pickle', 'wb') as file:
        pickle.dump({
            'A': A,
            'B': B,
            'cost': cost,
            'schedule_s_greedy': schedule_s_greedy_all,
            'cost_s_greedy': cost_s_greedy_all,
            's_greedy_mcmc': schedule_s_greedy_mcmc_all,
            'cost_s_greedy_mcmc': cost_s_greedy_mcmc_all
        }, file)
