import numpy as np

from typing import *
from scipy.linalg import qr
from copy import deepcopy
from random import random, sample

from .cost_function import CostFunction
from .utils import *

EPS = 1e-10
PRINT_DIGITS = 3

class Designer:

    ALGORITHMS = {'greedy-b', 'greedy-f', 'relax'}
    SPARSITY_TYPES = {'pw', 'avg'}

    def __init__(self, 
                 A: Union[list[np.ndarray], np.ndarray],
                 B: Union[list[np.ndarray], np.ndarray],
                 cost: CostFunction,
                 sparsity: int,
                 sparsity_type: str,
                 algo: str,
    ) -> None:
        self.A = A
        self.n = len(self.A) if isinstance(self.A, np.ndarray) else len(self.A[0])
        self.B = B
        self.m = len(self.B[0]) if isinstance(self.B, np.ndarray) else len(self.B[0][0])
        self.cost = cost
        self.s = sparsity
        if not sparsity_type in self.SPARSITY_TYPES:
            raise NotImplementedError('Sparsity constraint not implemented')
        
        if not algo in self.ALGORITHMS:
            raise NotImplementedError('Control design algorithm not implemented')
            
        self.sparsity_type = sparsity_type
        self.algo = algo
        

    def design(self) -> tuple[list[int], float]:
        if self.sparsity_type == 'pw':
            if self.algo == 'greedy-b':
                return self.greey_algo_backwards_pw()
            elif self.algo == 'greedy-f':
                return self.greey_algo_forward_pw()
            elif self.algo == 'relax':
                pass
            
        elif self.sparsity_type == 'avg':
            pass
    

    def greey_algo_backwards_pw(self) -> tuple[list[list[int]], float]:
        schedule_best = [None] * self.cost.h
        Bs = [deepcopy(self.B)] * self.cost.h if isinstance(self.B, np.ndarray) else deepcopy(self.B)
        A_curr = np.eye(self.n)
        col_space_contr_mat = None
        rk_contr_mat = 0
        ch_cand_dep = [None] * self.cost.h
        can_select = 0
        for k in range(self.cost.h):

            # iteration k optimizes the input channels applied at the (h-k)-th time step
            B_curr = self.B if isinstance(self.B, np.ndarray) else self.B[-1-k]
            A_prev = self.A if isinstance(self.A, np.ndarray) else self.A[-1-k]
            A_prod = np.matmul(A_curr, A_prev)
            
            # columns of B s.t. ColSpace{A^k B_s} complements ColSpace{A^(k+1)} but not ColSpace{A^k}
            im_B, B_cand = self.left_kernel(A_curr, B_curr, A_prod)

            # independent columns among found ones
            # these are needed for controllability
            _, ch_cand_ctrl_ind = self.independent_cols(B_cand)
            schedule_k = ch_cand_ctrl_ind

            # greedy selection of remaining channels
            if self.s > len(schedule_k):
                ch_cand = list(set(range(self.m)) - set(schedule_k))
                if rk_contr_mat < self.n:
                    
                    # priority to columns that increase rank of controllability matrix
                    _, ch_cand_ind = self.independent_cols(im_B[:, ch_cand], col_space_contr_mat, B_indep=True)
                    schedule_k = self.greedy_selection_k(k, B_curr, ch_cand_ind, schedule_k, deepcopy(Bs), eps=EPS)

                    # update column space of controllability matrix
                    col_space_contr_mat = np.hstack([im_B[:, schedule_k], col_space_contr_mat]) if col_space_contr_mat is not None else im_B[:, schedule_k]
                    rk_contr_mat += len(schedule_k)

                # store remaining candidate columns, if budget not exhausted
                if self.s > len(schedule_k):
                    can_select += self.s - len(schedule_k)
                    ch_cand_dep[k] = list(set(ch_cand) - set(ch_cand_ind))

            # store schedule
            schedule_best[-1-k] = deepcopy(schedule_k)
            Bs[-1-k] = deepcopy(B_curr[:, schedule_k])

            # update open-loop dynamics for next input selection
            A_curr = A_prod

        cost_best = self.cost.compute(self.A, Bs)

        # greedy selection of remaining columns across whole controllability matrix, if budget not exhausted
        Bs_cand = deepcopy(Bs)
        while can_select > 0:
            cand_best = None
            for k in range(self.cost.h):
                B_curr = self.B if isinstance(self.B, np.ndarray) else self.B[-1-k]
                for cand in ch_cand_dep[k]:
                    Bs_cand[-1-k] = np.hstack([Bs[-1-k], B_curr[:, cand]])
                    cost_cand = self.cost.compute(self.A, Bs_cand)
                    if cost_cand < cost_best:
                        cand_best = [k, cand]
                        cost_best = cost_cand

            if cand_best is not None:
                k, cand = cand_best[0], cand_best[1]
                schedule_best[-1-k].append(cand)
                ch_cand_dep[k].remove(cand)
                can_select -= 1

            else:
                break
        
        return schedule_best, cost_best


    def greey_algo_forward_pw(self) -> tuple[list[list[int]], float]:
        schedule_best = [None] * self.cost.h
        Bs = [deepcopy(self.B)] * self.cost.h if isinstance(self.B, np.ndarray) else deepcopy(self.B)
        col_space_contr_mat = None
        rk_contr_mat = 0
        ch_cand_dep = [None] * self.cost.h
        can_select = 0

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
                im_B, B_cand = self.left_kernel(A_all[k], B_curr, A_all[k-1])

                # independent columns among found ones
                # these are needed for controllability
                _, ch_cand_ctrl_ind = self.independent_cols(B_cand)
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
                    _, ch_cand_ind = self.independent_cols(im_B[:, ch_cand], col_space_contr_mat, B_indep=True)
                    schedule_k = self.greedy_selection_k(-1-k, B_curr, ch_cand_ind, schedule_k, deepcopy(Bs), EPS)

                    # update column space of controllability matrix
                    col_space_contr_mat = np.hstack([im_B[:, schedule_k], col_space_contr_mat]) if col_space_contr_mat is not None else im_B[:, schedule_k]
                    rk_contr_mat += len(schedule_k)

                # store remaining candidate columns, if budget not exhausted
                if self.s > len(schedule_k):
                    can_select += self.s - len(schedule_k)
                    ch_cand_dep[k] = list(set(ch_cand) - set(ch_cand_ind))

            # store schedule
            schedule_best[k] = deepcopy(schedule_k)
            Bs[k] = deepcopy(B_curr[:, schedule_k])

        cost_best = self.cost.compute(self.A, Bs)

        # greedy selection of remaining columns across whole controllability matrix, if budget not exhausted
        Bs_cand = deepcopy(Bs)
        while can_select > 0:
            cand_best = None
            for k in range(self.cost.h):
                B_curr = self.B if isinstance(self.B, np.ndarray) else self.B[k]
                for cand in ch_cand_dep[k]:
                    Bs_cand[k] = np.hstack([Bs[k], B_curr[:, cand]])
                    cost_cand = self.cost.compute(self.A, Bs_cand)
                    if cost_cand < cost_best:
                        cand_best = [k, cand]
                        cost_best = cost_cand
                        
            if cand_best is not None:
                k, cand = cand_best[0], cand_best[1]
                schedule_best[k].append(cand)
                ch_cand_dep[k].remove(cand)
                can_select -= 1

            else:
                break
        
        return schedule_best, cost_best
    

    def greedy_selection_k(self,
                            k: int,
                            B_curr: np.ndarray,
                            ch_cand: list[int],
                            schedule_k: list[int],
                            Bs_cand: list[np.ndarray],
                            eps: float = 0.
    ):
        if len(schedule_k):
            Bs_cand[-1-k] = B_curr[:, schedule_k]
            cost_curr_best = self.cost.compute(self.A, Bs_cand)

        else:
            cost_curr_best = np.inf

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

            else:
                break

        print(f'Cost at iter {k}:', truncate_float(cost_curr_best, PRINT_DIGITS))

        return schedule_k
    

    def MCMC(self,
             t_init: float = 1.,
             t_min: float = 1e-7,
             a: float = .1,
             it_max: int = 1000,
             schedule: list[list[int]] = None
    ) -> tuple[list[list[int]], float]:
        random_schedule = False
        if schedule is None:
            random_schedule = True
            schedule = [None] * self.cost.h
            for k in range(self.cost.h):
                schedule[k] = sample(range(self.m, self.s))

        Bs = [None] * self.cost.h
        for k in range(self.cost.h):
            Bs[k] = deepcopy(self.B[:, schedule[k]]) if isinstance(self.B, np.ndarray) else deepcopy(self.B[k][:, schedule[k]])

        t = t_init
        all_ch = self.cost.h * self.s
        cost_best = self.cost.compute(self.A, Bs, EPS) if random_schedule else self.cost.compute(self.A, Bs)
        while t > t_min:
            for _ in range(it_max):

                # select column in current schedule uniformly at random
                test = sample(range(all_ch), 1)[0]
                
                # sample candidate column for same time step
                k = test // self.s
                B_curr = self.B if isinstance(self.B, np.ndarray) else self.B[k]
                cand_k = list(set(range(self.m)) - set(schedule[k]))
                cand = sample(cand_k, 1)[0]
                Bs[k][:, test] = B_curr[:, cand]
                cost_curr = self.cost.compute(self.A, Bs, EPS)

                # select candidate column according to MCMC rule
                if cost_curr < cost_best or random() < np.exp(-(cost_best - cost_curr) / t):
                    schedule[k].remove(test)
                    schedule[k].append(cand)
                    cost_best = cost_curr
                
                else:

                    # reset tested scheduled column in input matrix
                    Bs[k][:, cand] = B_curr[:, test]

            t *= a

        return schedule, cost_best
    

    def left_kernel(self, A_curr, B_curr, A_prod):

        # image of B through A^k
        im_B = np.matmul(A_curr, B_curr)
        im_B[abs(im_B) < EPS] = 0

        # remove columns in ker{A^k}
        im_B[:, list(np.where(~im_B.any(axis=0))[0])] = []

        # image of A^k B through A^(k+1).T
        im_AB = np.matmul(A_prod.T, im_B)
        im_AB[abs(im_AB) < EPS] = 0

        # keep columns not in ker{A^(k+1).T}
        ch_cand_ctrl = list(np.where(~im_AB.any(axis=0))[0])
        B_cand = B_curr[:, ch_cand_ctrl]
        return im_B, B_cand
    

    def independent_cols(self, 
                         A: np.ndarray, 
                         B: np.ndarray = None, 
                         B_indep: bool = False
    ) -> tuple[np.ndarray, list[int]]:
        '''
        Find linearly independent columns of given matrices.
        If B (resp., A) is None, linearly independent columns of A (resp., B) are returned.
        If B_indep is True,
        linearly independent columns of A that are also linearly independent from columns of B are returned.
        
        Params
        ----
        A: np.ndarray,
            first matrix
        B: np.ndarray,
            second matrix
        B_indep: bool,
            True if columns of B are linearly independent
        
        Returns
        ----
        ind_cols: np.ndarray,
            linearly independent columns of given matrices
        ind_cols_idx: list[int],
            indices of returned columns
        '''
        if A is None:
            return self.independent_cols(B, A)
        if B is None:
            R = qr(A, mode='r', check_finite=False)
            ind_col_idx = self.staircase(R[0])
            return A[:, ind_col_idx], ind_col_idx

        if B_indep:
            R = qr(np.hstack((B, A)), mode='r', check_finite=False)
            ind_col_idx = self.staircase(R[0])
            ind_col_idx_A = [idx - np.shape(B)[1] for idx in ind_col_idx if idx >= np.shape(B)[1]]
            return A[:, ind_col_idx_A], ind_col_idx_A
        
        else:
            R = qr(np.hstack((A, B)), mode='r', check_finite=False)
            ind_col_idx = self.staircase(R[0])
            ind_col_idx_A = [idx for idx in ind_col_idx if idx < np.shape(A)[1]]
            ind_col_idx_B = [idx - np.shape(A)[1] for idx in list(set(ind_col_idx) - set(ind_col_idx_A))]
            return np.hstack(
                (A[:, ind_col_idx_A],
                 B[:, ind_col_idx_B])
            ), ind_col_idx
        
    
    def staircase(self, R: np.ndarray, tol: float = EPS) -> list[int]:
        '''
        Finds indices of linearly independent columns from QR decomposition
        
        Params
        ----
        R: np.ndarray,
            matrix R obtained from QR decomposition
        tol: float,
            tolerance for zeros
        
        Returns
        ----
        ind_col_idx: list[int],
            indices of linearly independent columns 
        '''
        ind_col_idx = []
        last_zero_row = 0
        for col_idx in range(np.shape(R)[1]):
            if abs(R[last_zero_row, col_idx]) > tol:
                ind_col_idx.append(col_idx)
                last_zero_row += 1
                if last_zero_row == np.shape(R)[0]:
                    break

        return ind_col_idx