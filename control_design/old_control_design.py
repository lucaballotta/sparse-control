import numpy as np

from typing import *
from copy import deepcopy
from random import random, sample

from .cost_function import CostFunction
from .utils import *

EPS = 1e-10
PRINT_DIGITS = 3

class Designer:

    ALGORITHMS = {'greedy', 'greedy-b', 'greedy-f', 'mcmc', 'relax'}

    def __init__(self, 
                 A: Union[List[np.ndarray], np.ndarray],
                 B: Union[List[np.ndarray], np.ndarray],
                 sparsity: int,
                 cost: CostFunction,
                 algo: str,
    ) -> None:
        self.A = A
        self.n = len(self.A) if isinstance(self.A, np.ndarray) else len(self.A[0])
        self.B = B
        self.m = len(self.B[0]) if isinstance(self.B, np.ndarray) else len(self.B[0][0])
        self.cost = cost
        if sparsity <= 0:
            raise ValueError('InputError: sparsity value must be positive')
        else:
            self.s = sparsity
        
        if not algo in self.ALGORITHMS:
            raise NotImplementedError('Requested algorithm not implemented')
        else:
            self.algo = algo
        

    def set_sparsity(self, sparsity: int):
        if sparsity <= 0:
            raise ValueError('InputError: sparsity value must be positive')
        
        self.s = sparsity


    def set_algo(self, algo: str):
        if not algo in self.ALGORITHMS:
            raise NotImplementedError('Requested algorithm not implemented')
            
        self.algo = algo


    def design(self, *args, **kwargs) -> Tuple[List[List[int]], float]:
        if self.algo == 'greedy':
            return self.greedy(*args, **kwargs)
        if self.algo == 'greedy-b':
            return self.greedy_backwards(*args, **kwargs)
        elif self.algo == 'greedy-f':
            return self.greedy_forward(*args, **kwargs)
        elif self.algo == 'mcmc':
            return self.mcmc(*args, **kwargs)
        elif self.algo == 'relax':
            pass
    

    def greedy(self, 
               ch_cand: List[List[int]] = None,
               schedule: List[List[int]] = None, 
               eps: float = 0.
    ) -> Tuple[List[List[int]], float]:
        if ch_cand is None:
            ch_cand = [list(range(self.m)) for _ in range(self.cost.h)]

        if schedule is None:
            Bs = [np.zeros((self.n, 1)) for _ in range(self.cost.h)]
            schedule =  [[] for _ in range(self.cost.h)]
            cost_best = np.inf

        else:
            if isinstance(self.B, np.ndarray):
                Bs = [self.B[:, schedule_k] for schedule_k in schedule]
            else:
                Bs = [self.B[k][:, schedule_k] for k, schedule_k in enumerate(schedule)]

            cost_best = self.cost.compute(self.A, Bs)

        cand_times = [k for k in range(self.cost.h) if len(schedule[k]) < self.s]
        while len(cand_times) > 0:
            cand_best = None
            for k in cand_times:
                B_curr = self.B if isinstance(self.B, np.ndarray) else self.B[k]
                for cand in ch_cand[k]:
                    Bs_cand = deepcopy(Bs)
                    Bs_cand[k] = np.hstack([Bs[k], np.reshape(B_curr[:, cand], (self.n, -1))]) if Bs[k].any() else np.reshape(B_curr[:, cand], (self.n, -1))
                    cost_cand = self.cost.compute_robust(self.A, Bs_cand, eps)
                    if cost_cand < cost_best:
                        cand_best = (k, cand, B_curr[:, cand])
                        cost_best = cost_cand
                        
            if cand_best is not None:
                k, cand, col = cand_best[0], cand_best[1], cand_best[2]
                schedule[k].append(cand)
                ch_cand[k].remove(cand)
                Bs[k] = np.hstack([Bs[k], np.reshape(col, (self.n, -1))]) if Bs[k].any() else np.reshape(col, (self.n, -1))
                if len(schedule[k]) == self.s:
                    cand_times.remove(k)

            else:
                break

        return schedule, cost_best
    

    def greedy_backwards(self, verbose: bool = False) -> Tuple[List[List[int]], float]:
        schedule_best = [None] * self.cost.h
        Bs = [self.B] * self.cost.h if isinstance(self.B, np.ndarray) else self.B
        A_curr = np.eye(self.n)
        col_space_contr_mat = None
        rk_contr_mat = 0
        ch_cand_dep = [None] * self.cost.h
        for k in range(self.cost.h):

            # iteration k optimizes the input channels applied at the (h-k)-th time step
            B_curr = self.B if isinstance(self.B, np.ndarray) else self.B[-1-k]
            A_prev = self.A if isinstance(self.A, np.ndarray) else self.A[-1-k]
            A_prod = np.matmul(A_curr, A_prev)
            
            # columns of B s.t. ColSpace{A^k B_s} complements ColSpace{A^(k+1)} but not ColSpace{A^k}
            im_B, ch_cand_ctrl = left_kernel(A_curr, B_curr, A_prod)

            # independent columns among found ones
            # these are needed for controllability
            _, ch_cand_ctrl_ind = independent_cols(B_curr[:, ch_cand_ctrl])
            schedule_k = ch_cand_ctrl_ind

            # greedy selection of remaining channels
            if self.s > len(schedule_k):
                ch_cand = list(set(range(self.m)) - set(schedule_k))
                if rk_contr_mat < self.n:
                    
                    # priority to columns that increase rank of controllability matrix
                    _, ch_cand_ind = independent_cols(im_B[:, ch_cand], col_space_contr_mat, B_indep=True)
                    schedule_k = self.greedy_k(k, B_curr, ch_cand_ind, schedule_k, Bs, EPS, verbose)

                    # update column space of controllability matrix
                    col_space_contr_mat = np.hstack([im_B[:, schedule_k], col_space_contr_mat]) if col_space_contr_mat is not None else im_B[:, schedule_k]
                    rk_contr_mat += len(schedule_k)

                # store remaining candidate columns, if budget not exhausted
                if self.s > len(schedule_k):
                    ch_cand_dep[k] = list(set(range(self.m)) - set(schedule_k))

            # store schedule
            schedule_best[-1-k] = deepcopy(schedule_k)

            # update open-loop dynamics for next input selection
            A_curr = A_prod

        # greedy selection of remaining columns across whole controllability matrix, if budget not exhausted
        schedule_best, cost_best = self.greedy(ch_cand_dep, schedule_best)
        
        return schedule_best, cost_best


    def greedy_forward(self, verbose: bool = False) -> Tuple[List[List[int]], float]:
        schedule_best = [None for _ in range(self.cost.h)]
        Bs = [self.B] * self.cost.h if isinstance(self.B, np.ndarray) else self.B
        col_space_contr_mat = None
        rk_contr_mat = 0
        ch_cand_dep = [None for _ in range(self.cost.h)]

        # pre-compute matrices applied to input channels
        A_all = [np.zeros(self.n)] * self.cost.h
        A_all[-1] = np.eye(self.n)
        A_curr = np.eye(self.n)
        for k in range(1, self.cost.h):
            A_prev = self.A if isinstance(self.A, np.ndarray) else self.A[-k]
            A_curr = np.matmul(A_curr, A_prev)

            # matrix for input at k-th time step stored at location k
            A_all[-1-k] = deepcopy(A_curr)

        for k in range(self.cost.h):

            # iteration k optimizes the input channels applied at the k-th time step
            B_curr = self.B if isinstance(self.B, np.ndarray) else self.B[k]
            if k > 0:
                
                # columns of B s.t. ColSpace{A^(h-k-1) B_s} complements ColSpace{A^(h-k)} but not ColSpace{A^(h-k-1)}
                im_B, ch_cand_ctrl = left_kernel(A_all[k], B_curr, A_all[k-1])

                # independent columns among found ones
                # these are needed for controllability
                _, ch_cand_ctrl_ind = independent_cols(B_curr[:, ch_cand_ctrl])
                schedule_k = ch_cand_ctrl_ind

            else:

                # image of B through A^(h-k-1)
                im_B = np.matmul(A_all[k], B_curr)
                im_B[abs(im_B) < EPS] = 0
                schedule_k = []

            # greedy selection of remaining channels
            if self.s > len(schedule_k):
                ch_cand = list(set(range(self.m)) - set(schedule_k))
                if rk_contr_mat < self.n:
                    
                    # priority to columns that increase rank of controllability matrix
                    _, ch_cand_ind = independent_cols(im_B[:, ch_cand], col_space_contr_mat, B_indep=True)
                    schedule_k = self.greedy_k(-1-k, B_curr, ch_cand_ind, schedule_k, Bs, EPS, verbose)

                    # update column space of controllability matrix
                    col_space_contr_mat = np.hstack([im_B[:, schedule_k], col_space_contr_mat]) if col_space_contr_mat is not None else im_B[:, schedule_k]
                    rk_contr_mat += len(schedule_k)

                # store remaining candidate columns, if budget not exhausted
                if self.s > len(schedule_k):
                    ch_cand_dep[k] = list(set(range(self.m)) - set(schedule_k))

            # store schedule
            schedule_best[k] = deepcopy(schedule_k)

        # greedy selection of remaining columns across whole controllability matrix, if budget not exhausted
        schedule_best, cost_best = self.greedy(ch_cand_dep, schedule_best)
        
        return schedule_best, cost_best


    def greedy_k(self,
                k: int,
                B_curr: np.ndarray,
                ch_cand: list[int],
                schedule_k: list[int],
                Bs: list[np.ndarray],
                eps: float = 0.,
                verbose: bool = False
    ) -> List[int]:
        if len(schedule_k):
            Bs[-1-k] = B_curr[:, schedule_k]
            cost_curr_best = self.cost.compute(self.A, Bs)

        else:
            cost_curr_best = np.inf

        Bs_cand = deepcopy(Bs)
        while len(schedule_k) < self.s:
            cand_best = None
            for cand in ch_cand:
                Bs_cand[-1-k] = B_curr[:, [*schedule_k, cand]]
                cost_cand = self.cost.compute(self.A, Bs_cand, eps)
                if cost_cand < cost_curr_best:
                    cand_best = cand
                    cost_curr_best = cost_cand
                    
            if cand_best is not None:
                schedule_k.append(cand_best)
                ch_cand.remove(cand_best)
                Bs[-1-k] = B_curr[:, schedule_k]

            else:
                break

        if verbose:
            print(f'Cost at iter {k}:', truncate_float(cost_curr_best, PRINT_DIGITS))

        return schedule_k
    

    def mcmc(self,
             t_init: float = 1.,
             t_min: float = 1e-7,
             a: float = .1,
             it_max: int = 5000,
             schedule: list[list[int]] = None,
             eps: float = 0.,
             check_rank: bool = False
    ) -> Tuple[List[List[int]], float]:
        random_schedule = False
        if schedule is None:
            random_schedule = True
            schedule = [None] * self.cost.h
            for k in range(self.cost.h):
                schedule[k] = sample(range(self.m), self.s)

        Bs = [None] * self.cost.h
        for k in range(self.cost.h):
            Bs[k] = deepcopy(self.B[:, schedule[k]]) if isinstance(self.B, np.ndarray) else deepcopy(self.B[k][:, schedule[k]])

        t = t_init
        all_col = self.cost.h * self.s
        cost_best = self.cost.compute_robust(self.A, Bs, eps) if random_schedule else self.cost.compute(self.A, Bs)
        if check_rank:
            rank = self.cost.get_contr_mat_rank()
            A_all = [np.zeros(self.n)] * self.cost.h
            A_all[-1] = np.eye(self.n)
            A_curr = np.eye(self.n)
            for k in range(1, self.cost.h):
                A_prev = self.A if isinstance(self.A, np.ndarray) else self.A[-k]
                A_curr = np.matmul(A_curr, A_prev)
                A_all[-1-k] = deepcopy(A_curr)

        while t > t_min:
            for _ in range(it_max):

                # select column in current schedule uniformly at random
                col = sample(range(all_col), 1)[0]
                k = col // self.s
                pos_k = col % self.s

                # sample candidate column for same time step                    
                B_curr = self.B if isinstance(self.B, np.ndarray) else self.B[k]
                cand_k = list(set(range(self.m)) - set(schedule[k]))
                cand = sample(cand_k, 1)[0]
                if check_rank:
                    _, ch_ctrl = left_kernel(A_all[k], B_curr, A_all[k-1])
                    _, ch_ctrl_ind = independent_cols(B_curr[:, ch_ctrl])
                    if pos_k in ch_ctrl_ind:
                        span_kernel = True
                        while cand not in ch_ctrl:
                            cand_k.remove(cand)
                            try:
                                cand = sample(cand_k, 1)[0]
                            except ValueError:
                                span_kernel = False
                                break

                        if not span_kernel:
                            continue

                Bs[k][:, pos_k] = B_curr[:, cand]
                if check_rank:
                    self.cost.update_gramian(self.A, Bs)
                    drop_rank = False
                    while rank > self.cost.get_contr_mat_rank():
                        cand_k.remove(cand)
                        try:
                            cand = sample(cand_k, 1)[0]
                            Bs[k][:, pos_k] = B_curr[:, cand]
                            self.cost.update_gramian(self.A, Bs)

                        except ValueError:
                            drop_rank = True
                            break

                    if drop_rank:
                        continue

                cost_curr = self.cost.compute_robust(self.A, Bs, eps)

                # select candidate column according to MCMC rule
                if cost_curr < cost_best or random() < np.exp(-(cost_curr - cost_best) / t):
                    schedule[k][pos_k] = cand
                    cost_best = cost_curr
                    if check_rank and rank < self.cost.get_contr_mat_rank():
                        rank = self.cost.get_contr_mat_rank()
                
                else:

                    # reset tested scheduled column in input matrix
                    Bs[k][:, pos_k] = B_curr[:, schedule[k][pos_k]]

            t *= a

        return schedule, cost_best
    