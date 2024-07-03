from curses import has_key
import datetime
import os.path
import pickle
from numpy.linalg import matrix_rank

from control_design.control_design import Designer
from control_design.cost_function import CostFunction

# import model matrices A and B
load_exp = True
save_result = True

if not load_exp:
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

    sparsity = max(len(A) - matrix_rank(A), 1)
    h = len(A)
    cost = 'tr-inv'

else:
    file_name = 'ex5_202407020602'
    with open('exp/' + file_name + '.pickle', 'rb') as f:
        vars = pickle.load(f) # type: dict
    
    A = vars['A'] # type: np.ndarray
    B = vars['B'] # type: np.ndarray
    h = len(A)
    cost = vars['cost'] # type: str
    sparsity = vars['s'] # type: int

# sparsity constraints
s_init = sparsity
s_step = 2
s_max = min(s_init + 10, len(B[0]) - s_step)
if load_exp:
    s_init += s_step

s_vec = range(s_init, s_max, s_step)

# initialize output vectors
schedule_s_greedy_all = dict.fromkeys(s_vec)
cost_s_greedy_all = dict.fromkeys(s_vec)
schedule_s_greedy_mcmc_all = dict.fromkeys(s_vec)
cost_s_greedy_mcmc_all = dict.fromkeys(s_vec)

if load_exp:
    schedule_s_greedy_all[sparsity] = vars['s_greedy'][0]
    cost_s_greedy_all[sparsity] = vars['s_greedy'][1]
    if 's_greedy_mcmc' in vars.keys():
        schedule_s_greedy_mcmc_all[sparsity] = vars['s_greedy_mcmc'][0]
        cost_s_greedy_mcmc_all[sparsity] = vars['s_greedy_mcmc'][1]

# find sparsity schedules
cost_func = CostFunction(h, cost)
designer = Designer(A, B, sparsity, cost_func)

if 's_greedy_mcmc' not in vars.keys():
    
    # s-sparse greedy + MCMC
    print(f'sparsity: {sparsity} (MCMC)\n')
    designer.set_algo('mcmc')
    schedule_s_greedy_mcmc, cost_s_greedy_mcmc = designer.design(schedule=schedule_s_greedy_all[sparsity])
    schedule_s_greedy_mcmc_all[sparsity] = schedule_s_greedy_mcmc
    cost_s_greedy_mcmc_all[sparsity] = cost_s_greedy_mcmc

for s in s_vec:
    print(f'sparsity: {s}\n')
    designer.set_sparsity(s)

    # s-sparse greedy
    designer.set_algo('greedy-f')
    schedule_s_greedy, cost_s_greedy = designer.design()
    schedule_s_greedy = [schedule_k for schedule_k in schedule_s_greedy if len(schedule_k) > 0]
    cost_s_greedy_all[s] = cost_s_greedy
    schedule_s_greedy_all[s] = schedule_s_greedy

    # s-sparse greedy + MCMC
    designer.set_algo('mcmc')
    schedule_s_greedy_mcmc, cost_s_greedy_mcmc = designer.design(schedule=schedule_s_greedy)
    cost_s_greedy_mcmc_all[s] = cost_s_greedy_mcmc
    schedule_s_greedy_mcmc_all[s] = schedule_s_greedy_mcmc

if save_result:
    dir_name = 'exp'
    if not os.path.isdir(dir_name):
        os.mkdir(dir_name)

    file_name = file_name[:file_name.find('_')] + '_s_' + datetime.datetime.now().strftime('%Y%m%d%I%M')
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
