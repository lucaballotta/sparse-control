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
                 A: Union[list[np.ndarray], np.ndarray],
                 B: Union[list[np.ndarray], np.ndarray],
                 sparsity: int,
                 cost: CostFunction = 'tr-inv',
                 algo: str = 'greedy',
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


    def design(self, *args, **kwargs) -> tuple[list[int], float]:
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
               ch_cand: list[list[int]] = None,
               schedule: list[list[int]] = None, 
               eps: float = 0.,
               check_rank: bool = False,
               contr_mat: np.ndarray = None,
               rank_contr_mat: int = None,
               A_vec: list[np.ndarray] = None
    ):
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

            cost_best = self.cost.compute(self.A, Bs) if not check_rank else self.cost.compute_robust(self.A, Bs, eps)

        cand_times = [k for k in range(self.cost.h) if len(schedule[k]) < self.s]
        it = 1
        while len(cand_times) > 0:
            print(f'iteration {it} of first loop')
            cand_best = None
            for k in cand_times:
                print(f'candidate time {k}')
                B_curr = self.B if isinstance(self.B, np.ndarray) else self.B[k]
                for cand in ch_cand[k]:
                    print(f'candidate {cand}')
                    Bs_cand = deepcopy(Bs)
                    Bs_cand[k] = np.hstack([Bs[k], B_curr[:, [cand]]]) if Bs[k].any() else B_curr[:, [cand]]
                    cost_cand = self.cost.compute_robust(self.A, Bs_cand, eps)
                    if cost_cand < cost_best:
                        cand_best = (k, cand, B_curr[:, cand])
                        cost_best = cost_cand
                        
            if cand_best is not None:
                k, cand, col = cand_best[0], cand_best[1], np.reshape(cand_best[2], (self.n, -1))
                schedule[k].append(cand)
                ch_cand[k].remove(cand)
                Bs[k] = np.hstack([Bs[k], col]) if Bs[k].any() else col
                if len(schedule[k]) == self.s:
                    cand_times.remove(k)
                
                if check_rank:
                    im_col = np.reshape(np.matmul(A_vec[k], col), (self.n, -1))
                    contr_mat = np.hstack([contr_mat, im_col]) if contr_mat is not None else im_col
                    rank_contr_mat += 1
                    if rank_contr_mat == self.n:
                        break

                    for j in cand_times:
                        ch_cand_j_copy = deepcopy(ch_cand[j])
                        B_j = self.B if isinstance(self.B, np.ndarray) else self.B[j]
                        for cand_j in ch_cand_j_copy:
                            AB = np.matmul(A_vec[j], np.reshape(B_j[:, cand_j], (self.n, -1))) if A_vec[j] is not None else np.reshape(B_j[:, cand_j], (self.n, -1))
                            _, idx = independent_cols(
                                AB,
                                contr_mat,
                                B_indep=True
                            )
                            if not len(idx):
                                ch_cand[j].remove(cand_j)

            else:
                break

            it += 1
        
        self.cost.update_gramian(self.A, Bs)
        if self.cost.get_gramian_rank() < self.n:
            cost_best = np.inf
            
        return schedule, cost_best
    

    def greedy_backwards(self, verbose: bool = False) -> tuple[list[list[int]], float]:
        schedule_best = [None] * self.cost.h
        Bs = [self.B] * self.cost.h if isinstance(self.B, np.ndarray) else self.B
        A_curr = np.eye(self.n)
        col_space_contr_mat = None
        rk_contr_mat = 0
        ch_cand = [None for _ in range(self.cost.h)]
        for k in range(self.cost.h - 1):
            if rk_contr_mat < self.n:

                # iteration k optimizes the input channels applied at the (h-k)-th time step
                B_curr = self.B if isinstance(self.B, np.ndarray) else self.B[-1-k]
                A_prev = self.A if isinstance(self.A, np.ndarray) else self.A[-1-k]
                A_prod = np.matmul(A_curr, A_prev)
                
                # columns of B s.t. ColSpace{A^k B_s} complements ColSpace{A^(k+1)} but not ColSpace{A^k}
                im_AB, im_K, ch_cand_ker = left_kernel(A_curr, B_curr, A_prod)

                # greedily select independent columns among found ones
                # these are needed for controllability
                schedule_k = self.greedy_k(k, B_curr, ch_cand_ker, [], Bs, EPS, verbose=verbose, check_rank=True, rank_mat=im_K)

                # update column space of controllability matrix
                col_space_contr_mat = np.hstack([im_AB[:, schedule_k], col_space_contr_mat]) if col_space_contr_mat is not None else im_AB[:, schedule_k]
                rk_contr_mat += len(schedule_k)

            else:
                schedule_k = []

            # update open-loop dynamics for next input selection
            A_curr = A_prod

            # store schedule
            schedule_best[-1-k] = deepcopy(schedule_k)

            # store remaining columns, if budget not exhausted
            if self.s > len(schedule_k):
                ch_cand[-1-k] = list(set(range(self.m)) - set(schedule_k))

        # first input is unconstrained
        schedule_best[0] = []
        ch_cand[0] = list(range(self.m))
        
        # greedily select independent columns till controllability Gramian has full rank
        # pre-compute matrices applied to input channels
        A_all = [np.zeros(self.n)] * self.cost.h
        A_all[-1] = np.eye(self.n)
        A_curr = np.eye(self.n)
        for k in range(1, self.cost.h):
            A_prev = self.A if isinstance(self.A, np.ndarray) else self.A[-k]
            A_curr = np.matmul(A_curr, A_prev)

            # matrix for input at k-th time step stored at location k
            A_all[-1-k] = deepcopy(A_curr)

        schedule_best, cost_best = self.greedy(
            ch_cand,
            schedule_best,
            EPS,
            check_rank=True,
            contr_mat=col_space_contr_mat,
            rank_contr_mat=rk_contr_mat,
            A_vec=A_all
        )

        # greedy selection of remaining columns across whole controllability matrix, if budget not exhausted
        schedule_best, cost_best = self.greedy(ch_cand, schedule_best)
        
        return schedule_best, cost_best


    def greedy_forward(self, verbose: bool = False) -> tuple[list[list[int]], float]:
        schedule_best = [None for _ in range(self.cost.h)]
        Bs = [self.B] * self.cost.h if isinstance(self.B, np.ndarray) else self.B
        col_space_contr_mat = None
        rk_contr_mat = 0
        ch_cand = [None for _ in range(self.cost.h)]

        # pre-compute matrices applied to input channels
        A_all = [np.zeros(self.n)] * self.cost.h
        A_all[-1] = np.eye(self.n)
        A_curr = np.eye(self.n)
        for k in range(1, self.cost.h):
            A_prev = self.A if isinstance(self.A, np.ndarray) else self.A[-k]
            A_curr = np.matmul(A_curr, A_prev)

            # matrix for input at k-th time step stored at location k
            A_all[-1-k] = deepcopy(A_curr)

        # first input is unconstrained
        schedule_best[0] = []
        ch_cand[0] = list(range(self.m))
        for k in range(1, self.cost.h):
            if rk_contr_mat < self.n:
                
                # iteration k optimizes the input channels applied at the k-th time step
                B_curr = self.B if isinstance(self.B, np.ndarray) else self.B[k]
                                
                # columns of B s.t. ColSpace{A^(h-k-1) B_s} complements ColSpace{A^(h-k)}
                im_AB, im_K, ch_cand_ker = left_kernel(A_all[k], B_curr, A_all[k-1])

                # greedily select independent columns among found ones
                # these are needed for controllability
                schedule_k = self.greedy_k(-1-k, B_curr, ch_cand_ker, [], Bs, EPS, verbose=verbose, check_rank=True, rank_mat=im_K)
                
                # update column space of controllability matrix
                col_space_contr_mat = np.hstack([im_AB[:, schedule_k], col_space_contr_mat]) if col_space_contr_mat is not None else im_AB[:, schedule_k]
                rk_contr_mat += len(schedule_k)

            else:
                schedule_k = []

            # store schedule
            schedule_best[k] = deepcopy(schedule_k)

            # store remaining columns, if budget not exhausted
            if self.s > len(schedule_k):
                ch_cand[k] = list(set(range(self.m)) - set(schedule_k))

        # greedily select independent columns till controllability Gramian has full rank
        schedule_best, cost_best = self.greedy(
            ch_cand,
            schedule_best,
            EPS,
            check_rank=True,
            contr_mat=col_space_contr_mat,
            rank_contr_mat=rk_contr_mat,
            A_vec=A_all
        )

        # greedy selection of remaining columns across whole controllability matrix, if budget not exhausted
        ch_cand_dep = [list(set(range(self.m)) - set(schedule_k)) for schedule_k in schedule_best]
        schedule_best, cost_best = self.greedy(ch_cand_dep, schedule_best)
        
        return schedule_best, cost_best


    def greedy_k(self,
                k: int,
                B_curr: np.ndarray,
                ch_cand: list[int],
                schedule_k: list[int],
                Bs: list[np.ndarray],
                eps: float = 0.,
                verbose: bool = False,
                check_rank : bool = False,
                rank_mat: np.ndarray = None
    ):
        if len(schedule_k):
            Bs[-1-k] = B_curr[:, schedule_k]
            cost_curr_best = self.cost.compute(self.A, Bs)

        else:
            cost_curr_best = np.inf

        Bs_cand = deepcopy(Bs)
        iter = 1
        while len(schedule_k) < self.s:
            cand_best = None
            for cand in ch_cand:
                Bs_cand[-1-k] = B_curr[:, [*schedule_k, cand]]
                cost_cand = self.cost.compute_robust(self.A, Bs_cand, eps)
                if cost_cand < cost_curr_best:
                    cand_best = cand
                    cost_curr_best = cost_cand
                    
            if cand_best is not None:
                schedule_k.append(cand_best)
                ch_cand.remove(cand_best)
                Bs[-1-k] = B_curr[:, schedule_k]
                if check_rank:
                    ch_cand_copy = deepcopy(ch_cand)
                    for cand in ch_cand_copy:
                        _, idx = independent_cols(
                            rank_mat[:, [cand]],
                            rank_mat[:, schedule_k],
                            B_indep=True
                        )
                        if not len(idx):
                            ch_cand.remove(cand)

            else:
                break

            iter += 1

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
    ) -> tuple[list[list[int]], float]:
        if schedule is None:
            schedule = [None] * self.cost.h
            for k in range(self.cost.h):
                schedule[k] = sample(range(self.m), self.s)

        Bs = [None] * self.cost.h
        for k in range(self.cost.h):
            Bs[k] = deepcopy(self.B[:, schedule[k]]) if isinstance(self.B, np.ndarray) else deepcopy(self.B[k][:, schedule[k]])

        t = t_init
        all_col = self.cost.h * self.s
        cost_best = self.cost.compute_robust(self.A, Bs, eps)
        if check_rank:
            rank = self.cost.get_gramian_rank()
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
                    _, ch_ker, _ = left_kernel(A_all[k], B_curr, A_all[k-1])
                    if pos_k in ch_ker:
                        span_kernel = True
                        while cand not in ch_ker:
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
                    while rank > self.cost.get_gramian_rank():
                        cand_k.remove(cand)
                        try:
                            cand = sample(cand_k, 1)[0]
                            Bs[k][:, pos_k] = B_curr[:, cand]
                            self.cost.update_gramian(self.A, Bs)

                        except ValueError:
                            drop_rank = True
                            Bs[k][:, pos_k] = B_curr[:, schedule[k][pos_k]]
                            break

                    if drop_rank:
                        continue

                cost_curr = self.cost.compute_robust(self.A, Bs, eps)

                # select candidate column according to MCMC rule
                if cost_curr < cost_best or random() < np.exp(-(cost_curr - cost_best) / t):
                    schedule[k][pos_k] = cand
                    cost_best = cost_curr
                    if check_rank and rank < self.cost.get_gramian_rank():
                        rank = self.cost.get_gramian_rank()
                
                else:

                    # reset tested scheduled column in input matrix
                    Bs[k][:, pos_k] = B_curr[:, schedule[k][pos_k]]

            t *= a

        self.cost.update_gramian(self.A, Bs)
        if self.cost.get_gramian_rank() < self.n:
            cost_best = np.inf

        return schedule, cost_best
    