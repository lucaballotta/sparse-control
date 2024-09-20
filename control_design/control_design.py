import numpy as np
import warnings

from typing import *
from types import NoneType
from copy import deepcopy
from random import random, sample

from .cost_function import CostFunction
from .utils import *

EPS = 1e-10
PRINT_DIGITS = 3

class Designer:
    r'''
    Class for s-sparse actuator schedule design of an LTI system of the form

    x(t+1) = A x(t) + B u(t)

    Attributes
    ----------
    A : np.ndarray
        State matrix.
    B : np.ndarray
        Input-state matrix.
    n : int
        Dimension of state x(t).
    m : int
        Dimension of control input u(t).
    cost : str
        Energy-related cost function to be minimized.
    sparsity : int
        Maximal number of input channels that can be scheduled at each time.
    algo: str
        Design algorithm.

    Methods
    --------
    set_sparsity

    set_cost
    
    set_algo
    
    design
    
    greedy
    
    s_greedy
    
    s_greedy_backwards
    
    greedy_k
    
    mcmc

    '''

    ALL_ALGO = {'greedy', 's-greedy', 's-greedy-b', 'mcmc'}

    def __init__(
            self, 
            A: Union[List[np.ndarray], np.ndarray],
            B: Union[List[np.ndarray], np.ndarray],
            sparsity: int,
            cost: CostFunction = 'tr-inv',
            algo: str = 's-greedy',
    ) -> None:
        self.A = A
        self.n = len(self.A) if isinstance(self.A, np.ndarray) else len(self.A[0])
        self.B = B
        self.m = len(self.B[0]) if isinstance(self.B, np.ndarray) else len(self.B[0][0])
        self.cost = cost
        self.s_min = max(1, self.n - np.linalg.matrix_rank(self.A))
        if sparsity < self.s_min:
            raise ValueError(f'InputError: sparsity must be at least{self.s_min}')
        else:
            self.s = sparsity
        
        if not algo in self.ALL_ALGO:
            raise NotImplementedError('Requested algorithm not implemented')
        else:
            self.algo = algo
        

    def set_sparsity(self, sparsity: int):
        if sparsity <= 0:
            raise ValueError(f'InputError: sparsity value must be at least{self.s_min}')
        
        self.s = sparsity


    def set_cost(self, cost: CostFunction):
        self.cost = cost


    def set_algo(self, algo: str):
        if not algo in self.ALL_ALGO:
            warnings.warn('Requested algorithm not implemented, setting default (greedy)', 
                          category=UserWarning)
            algo = 'greedy'

        self.algo = algo


    def design(
            self, 
            *args, 
            **kwargs
    ) -> Tuple[List[List[int]], float]:
        '''
        Runs the design algorithm in self.algo with cost function in self.cost

        Parameters
        ----
        *args, **kwargs
            Arguments required by design algorithm.

        Returns
        ---
        schedule: list[list[int]]
            The actuator schedule.
        
        cost_best: float
            The cost of the actuator schedule.

        '''
        if self.algo == 'greedy':
            return self.greedy(*args, **kwargs)
        elif self.algo == 's-greedy':
            return self.s_greedy(*args, **kwargs)
        elif self.algo == 's-greedy-b':
            return self.s_greedy_backwards(*args, **kwargs)
        elif self.algo == 'mcmc':
            return self.mcmc(*args, **kwargs)
    

    def greedy(
            self, 
            ch_cand: Union[List[List[int]], NoneType] = None,
            schedule: Union[List[List[int]], NoneType] = None, 
            eps: float = 0.,
            check_rank: bool = False,
            contr_mat: Union[np.ndarray, NoneType] = None,
            rank_contr_mat: Union[int, NoneType] = None,
            A_vec: Union[List[np.ndarray], NoneType] = None
    ) -> Tuple[List[List[int]], float]:
        '''
        Runs the naive greedy selection algorithm

        Parameters
        ----
        ch_cand: list[list[int]], optional
            Candidate channels for each time instant. 
            If None, all channels are selected
        schedule: list[list[int]], optional
            Initial partially filled schedule.
            If None, the schedule is initialized empty.
        eps: float, optional
            Slack parameter to compute the cost.
        check_rank: bool, optional
            If True, candidate channels that do not increase the rank of the controllability Gramian
            are removed after each selection
        contr_mat: np.ndarray, optional
            Reachability matrix with the selected schedule.
            Not used if check_rank is False 
        rank_contr_mat: int, optional
            Rank of controllability Gramian.
            Not used if check_rank is False
        A_vec: list[np.ndarray], optional
            List with pre-computed matrix powers of A.

        Returns
        ---
        schedule: list[list[int]]
            The actuator schedule
        
        cost_best: float
            The cost of the actuator schedule
        '''
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

            cost_best = self.cost.compute(self.A, Bs) if not check_rank else np.inf

        cand_times = [k for k in range(self.cost.h) if len(schedule[k]) < self.s]
        if check_rank:
            for j in cand_times:
                ch_cand_j_copy = deepcopy(ch_cand[j])
                B_j = self.B if isinstance(self.B, np.ndarray) else self.B[j]
                for cand_j in ch_cand_j_copy:
                    AB = np.matmul(A_vec[j], B_j[:, [cand_j]]) if A_vec[j] is not None else B_j[:, [cand_j]]
                    _, idx = independent_cols(AB, contr_mat, B_indep=True)
                    if not len(idx):
                        ch_cand[j].remove(cand_j)

            cand_times = [j for j in cand_times if len(ch_cand[j])]

        it = 1
        counter_no_improve = 0
        counter_no_improve_max = 0
        while len(cand_times) > 0:
            cand_best = None
            for k in cand_times:
                B_curr = self.B if isinstance(self.B, np.ndarray) else self.B[k]
                for cand in ch_cand[k]:
                    Bs_cand = deepcopy(Bs)
                    Bs_cand[k] = np.hstack([Bs[k], B_curr[:, [cand]]]) if Bs[k].any() else B_curr[:, [cand]]
                    cost_cand = self.cost.compute_robust(self.A, Bs_cand, eps)
                    if cost_cand < cost_best:
                        cand_best = (k, cand, B_curr[:, [cand]])
                        cost_best = cost_cand
                        
            if cand_best is not None:
                counter_no_improve = 0
                k, cand, col = cand_best[0], cand_best[1], cand_best[2]
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
                            AB = np.matmul(A_vec[j], B_j[:, [cand_j]]) if A_vec[j] is not None else B_j[:, [cand_j]]
                            _, idx = independent_cols(AB, contr_mat, B_indep=True)
                            if not len(idx):
                                ch_cand[j].remove(cand_j)

                    cand_times = [j for j in cand_times if len(ch_cand[j])]
                    cost_best = np.inf

            else:
                print('attempt random selection')
                if counter_no_improve == counter_no_improve_max:
                    break

                counter_no_improve += 1

                # choose randomly
                k = sample(cand_times, 1)[0]
                cand = sample(ch_cand[k], 1)[0]
                schedule[k].append(cand)
                ch_cand[k].remove(cand)
                B_curr = self.B if isinstance(self.B, np.ndarray) else self.B[k]
                Bs[k] = np.hstack([Bs[k], B_curr[:, [cand]]]) if Bs[k].any() else B_curr[:, [cand]]
                if len(schedule[k]) == self.s:
                    cand_times.remove(k)

            it += 1
        
        self.cost.update_gramian(self.A, Bs)
        if self.cost.get_contr_mat_rank() < self.n:
            cost_best = np.inf
            
        return schedule, cost_best
    

    def s_greedy(self) -> Tuple[List[List[int]], float]:
        '''
        Runs the s-sparse greedy selection algorithm.

        Returns
        ---
        schedule: list[list[int]]
            The actuator schedule.
        
        cost_best: float
            The cost of the actuator schedule.
        '''
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
            if rk_contr_mat < self.n and np.linalg.matrix_rank(A_all[k-1]) < self.n:
                
                # iteration k optimizes the input channels applied at the k-th time step
                B_curr = self.B if isinstance(self.B, np.ndarray) else self.B[k]

                # columns of B s.t. ColSpace{A^(h-k-1) B_s} complements ColSpace{A^(h-k)}
                im_AB, im_K, ch_cand_ker = left_kernel(A_all[k], B_curr, A_all[k-1])

                # greedily select independent columns among found ones
                # these are needed for controllability
                schedule_k = self.greedy_k(-1-k, B_curr, ch_cand_ker, Bs, EPS, im_K)
                
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
        if cost_best == np.inf:
            raise Warning('System is uncontrollable after rank-aware channel selection')
        else:
            print('System is controllable, improving cost')

        # greedy selection of remaining columns across whole controllability matrix, if budget not exhausted
        ch_cand_dep = [list(set(range(self.m)) - set(schedule_k)) for schedule_k in schedule_best]
        schedule_best, cost_best = self.greedy(ch_cand_dep, schedule_best)
        
        return schedule_best, cost_best
    

    def s_greedy_backwards(self) -> Tuple[List[List[int]], float]:
        '''
        Runs the s-sparse greedy selection algorithm backwards,
        running the first loop that selects the input channels that span the kernels of A^k from k = h-1 to k = 0.

        Returns
        ---
        schedule: list[list[int]]
            The actuator schedule.
        
        cost_best: float
            The cost of the actuator schedule.
        '''
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
                schedule_k = self.greedy_k(k, B_curr, ch_cand_ker, Bs, EPS, im_K)

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


    def greedy_k(
            self,
            k: int,
            B_curr: np.ndarray,
            ch_cand: list[int],
            Bs: list[np.ndarray],
            eps: float,
            rank_mat: np.ndarray
    ) -> List[int]:
        '''
        Selects columns of B that span the kernels of matrix powers A^k.

        Parameters
        ----
        k: int
            Exponent of matrix power A^k.
        B_curr: np.ndarray
            B matrix at step k.
        ch_cand: list[int]
            Candidate input channels.
        Bs: list[np.ndarray]
            Columns of B selected by the schedule.
        eps: float
            Slack parameter to compute the cost.
        rank_mat: int
            Rank of controllability Gramian.

        Returns
        ----
        schedule_k: list[int]
            Channels scheduled at time k.
        '''
        cost_curr_best = np.inf
        Bs_cand = deepcopy(Bs)
        schedule_k = []
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

        return schedule_k
    

    def mcmc(
            self,
            t_init: float = 1.,
            t_min: float = 1e-7,
            a: float = .1,
            it_max: int = 5000,
            schedule: Union[List[List[int]], NoneType] = None,
            eps: float = 0.,
            check_rank: bool = False
    ) -> Tuple[List[List[int]], float]:
        '''
        Runs s-sparse MCMC.

        Parameters
        ----
        t_init: float, optional
            Initial temperature parameter.
        t_min: float, optional
            Minimum temperature parameter.
        a: float, optional
            Multiplicative stepsize of temperature parameter.
        it_max: int, optional
            Number of iterations run with each T.
        schedule: list[list[int]], optional
            Initial schedule.
            If None, the schedule is initialized random.
        eps: float, optional
            Slack parameter to compute cost function.
        check_rank: bool, optional
            If True, the rank of the controllability Gramian is checked at each selection,
            until the candidate sample does not decrease the rank.

        Returns
        ---
        schedule: list[list[int]]
            The actuator schedule.
        
        cost_best: float
            The cost of the actuator schedule.

        '''
        if schedule is None:
            schedule = [None] * self.cost.h
            for k in range(self.cost.h):
                schedule[k] = sample(range(self.m), self.s)

        Bs = [None] * len(schedule)
        for k in range(len(schedule)):
            Bs[k] = deepcopy(self.B[:, schedule[k]]) if isinstance(self.B, np.ndarray) else deepcopy(self.B[k][:, schedule[k]])

        t = t_init
        all_col = np.vstack([[[k, ch_k] for ch_k in schedule_k] for k, schedule_k in enumerate(schedule) if len(schedule_k)]).tolist() #type: list
        cost_best = self.cost.compute_robust(self.A, Bs, eps)
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
                col = all_col[sample(range(len(all_col)), 1)[0]]
                k, ch_k = col[0], col[1]

                # sample candidate column for same time step                    
                B_curr = self.B if isinstance(self.B, np.ndarray) else self.B[k]
                cand_k = list(set(range(self.m)) - set(schedule[k]))
                cand = sample(cand_k, 1)[0]
                if check_rank:
                    _, _, ch_ker = left_kernel(A_all[k], B_curr, A_all[k-1])
                    if ch_k in ch_ker:
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
                
                cand_sch = deepcopy(schedule[k])  #type: list
                cand_sch.remove(ch_k)
                cand_sch.append(cand)
                Bs[k] = B_curr[:, cand_sch]
                if check_rank:
                    self.cost.update_gramian(self.A, Bs)
                    drop_rank = False
                    while rank > self.cost.get_contr_mat_rank():
                        cand_k.remove(cand)
                        cand_sch.remove(cand)
                        try:
                            cand = sample(cand_k, 1)[0]
                            cand_sch.append(cand)
                            Bs[k] = B_curr[:, cand_sch]
                            self.cost.update_gramian(self.A, Bs)

                        except ValueError:
                            drop_rank = True
                            Bs[k] = B_curr[:, schedule[k]]
                            break

                    if drop_rank:
                        continue

                cost_curr = self.cost.compute_robust(self.A, Bs, eps)

                # select candidate column according to MCMC rule
                if cost_curr < cost_best or random() < np.exp(-(cost_curr - cost_best) / t):
                    schedule[k] = deepcopy(cand_sch)
                    cost_best = cost_curr
                    if check_rank and rank < self.cost.get_contr_mat_rank():
                        rank = self.cost.get_contr_mat_rank()

                    all_col.remove(col)
                    all_col.append([k, cand])
                
                else:

                    # reset tested scheduled column in input matrix
                    Bs[k] = B_curr[:, schedule[k]]

            t *= a

        self.cost.update_gramian(self.A, Bs)
        if self.cost.get_contr_mat_rank() < self.n:
            cost_best = np.inf

        return schedule, cost_best
    
