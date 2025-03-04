import datetime
import os.path
import pickle
import numpy as np
import control as ct
from numpy.linalg import matrix_rank

from examples.ex6 import A, B
from control_design.control_design import Designer
from control_design.cost_function import CostFunction
from control_design.utils import get_q_redundant_controllable

import os
os.system('cls||clear')  # Clears the terminal, FROM: https://stackoverflow.com/questions/62742102/how-do-i-automatically-clear-the-terminal-in-vscode-before-execution-of-script  # nopep8

# to save results 
save_result = False

# import model matrices A and B
# A = np.array([[0.1,   0,   0,   0, 0, 0, 0, -7/2],
#               [0,   0.2,   0,   0, 0, 0, 0, -3],
#               [0,   0,   0.3,   0, 0, 0, 0, -5/2],
#               [3/4, 1/2, 0,   0.4, 0, 0, 0, 13/8],
#               [0,   3/4, 1/2, 0, 0.5, 0, 0, 11/8],
#               [5/4, 0,   3/4, 0, 0, 0.6, 0, 3/2],
#               [3/2, 5/4, 1,   0, 0, 0, 0.7, 9/4],
#               [0,   0,   0,   0, 0, 0, 0, 0.8]
#               ])
# B = np.eye(8)

# A = np.array([[0, 1], [-0.5, 0.2]])
# B = np.eye(2)

# A = np.array([[0.6, 1, 0], [0, -0.5, -0.5], [0, 0, 0.2]])
# B = np.eye(3)

# A = np.array([[-8, 0, -2], [0, -2, -8], [7, 0, -3]])
# B = np.eye(3)
# sys = ct.ss(A, B, np.eye(3), np.zeros((3, 3)))
# sys_d = sys.sample(True)
# A, B = sys_d.A, sys_d.B

# Get the dimensions of the system
n_x, n_u = A.shape[0], B.shape[1]

h = len(A)
cost, cost_bw, gram_type_bw = 'logdet', 'logdet', 'inf-cyclic'

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
cost_func, cost_func_bw = CostFunction(h, cost), CostFunction(h, cost_bw)
designer = Designer(A, B, s_init, cost_func)

# TEMP
s_vec = [s_init]

for s in s_vec:
    #: Get q-redundant controllable
    q_redundant_controllable = get_q_redundant_controllable(A, B)

    print(f"====== n_x = {A.shape[0]}, sparsity: {s} (static actuators needed = {A.shape[0] - q_redundant_controllable}) ======")
    designer.set_sparsity(s)

    # Forward greedy
    designer.set_algo('s-greedy')
    try:
        # Compute the schedule
        schedule_s_greedy, cost_s_greedy = designer.design()
        schedule_s_greedy = [schedule_k for schedule_k in schedule_s_greedy if len(schedule_k) > 0]
        cost_s_greedy_all[s] = cost_s_greedy
        schedule_s_greedy_all[s] = schedule_s_greedy

        # Check if all permutations are s-sparse controllable
        try:
            S_s_greedy, shift_ctrb = np.array([[1 if i in sublist else 0 for i in range(n_u)] for sublist in schedule_s_greedy]).T, True
            for idx_col in range(h):
                Psi = np.column_stack([np.linalg.matrix_power(A, h - k) @ B @ np.diag(S_s_greedy[:, k]) for k in range(h)])
                if matrix_rank(Psi) < n_x:
                    shift_ctrb = False
                    break
        except IndexError:
            shift_ctrb = None
    except Warning:
        schedule_s_greedy, cost_s_greedy, shift_ctrb = [], np.inf, None

    print(f"\n------ forwards greedy ------")
    print(f"schedule:\n{[sorted(sc) for sc in schedule_s_greedy]}")
    print(f'cost: {np.round(cost_s_greedy, 3) + 0} | {cost} (shift-invariant: {shift_ctrb})\n')

    # Backward greedy
    designer_bw = Designer(A, B, s, cost_func_bw)
    designer_bw.set_algo('backward-greedy')
    designer_bw.gram_type = gram_type_bw
    schedule_greedy_backward, cost_greedy_backward = designer_bw.design()
    
    print(f"------ backwards greedy ------")
    print(f"schedule:\n{[sorted(sc) for sc in schedule_greedy_backward]}")
    print(f'cost: {np.round(cost_greedy_backward, 3) + 0} | {cost_bw}, {gram_type_bw} (controllable: {np.linalg.matrix_rank(np.column_stack([np.linalg.matrix_power(A, k) @ B[:, sorted(schedule_greedy_backward[k])] for k in range(h)])) == A.shape[0]})\n')

    # s-sparse MCMC with warm start
    designer.set_algo('mcmc')
    try:
        schedule_s_greedy_mcmc, cost_s_greedy_mcmc = designer.design(schedule=schedule_s_greedy)
        cost_s_greedy_mcmc_all[s] = cost_s_greedy_mcmc
        schedule_s_greedy_mcmc_all[s] = schedule_s_greedy_mcmc
    except ValueError:
        schedule_s_greedy_mcmc, cost_s_greedy_mcmc = [], np.inf

    print(f"\n------ forwards greedy + MCMC ------")
    print(f"schedule:\n{[sorted(sc) for sc in schedule_s_greedy_mcmc]}")
    print(f'cost: {np.round(cost_s_greedy_mcmc, 3) + 0} | {cost}\n')


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
