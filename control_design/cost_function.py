from types import NoneType
import numpy as np
import scipy
import warnings

from typing import *
from matplotlib.pylab import LinAlgError

from .utils import fxn

EPS = 1e-10
COST_FUNCTIONS = {'logdet', 'tr', 'tr-inv', 'lambda-min'}

class CostFunction:
    r'''
    Class for computation of cost function used in control design problem.

    Attributes
    ----
    cost_func: str
        Function of the controllability Gramian.
    h: int
        Time horizon.
    _contr_mat: np.ndarray
        Reachability matrix of actuator schedule.
    _W: np.ndarray
        controllability Gramian of actuator schedule.

    Methods
    ----
    get_gramian

    get_contr_mat_rank

    compute

    compute_robust
    
    log_det_cost

    trace_cost

    trace_inv_cost

    lambda_min_cost

    update_gramian

    _add_to_gramian

    '''

    def __init__(self, h: int, cost_func: str) -> None:
        if cost_func not in COST_FUNCTIONS:
            raise NotImplementedError('Cost function not implemented')
        
        if h <= 0:
            raise ValueError('InputError: horizon h must be greater than zero')
        
        self.cost_func = cost_func
        self.h = h
        self._contr_mat = None
        self._W = None
            
        
    def get_gramian(self) -> np.ndarray:
        return self._W
    

    def get_contr_mat_rank(self) -> float:
        return np.linalg.matrix_rank(self._contr_mat)
    

    def compute(
        self,
        A: Union[np.ndarray, List[np.ndarray]],
        B: Union[np.ndarray, List[np.ndarray]],
        eps: float = 0.
    ) -> float:
        '''
        Evaluates the cost function.

        Parameters
        ----
        A: np.ndarray
            State matrix A.
        B: np.ndarray or list[np.ndarray]
            Input-state matrix B or list of submatrices of B according to actuator schedule,
            where the index in the list corresponds to the time instant.
        eps: float, optional
            Slack to compute the cost function.

        Returns
        ----
        cost: float,
            the value of the cost function.
        '''
        self.update_gramian(A, B)
        if self.cost_func == 'logdet':
            return self.log_det_cost(eps)
        elif self.cost_func == 'tr':
            return self.trace_cost()
        elif self.cost_func == 'tr-inv':
            return self.trace_inv_cost(eps)
        elif self.cost_func == 'lambda-min':
            return self.lambda_min_cost(eps)
        

    def compute_robust(self,
                       A: Union[np.ndarray, List[np.ndarray]],
                       B: Union[np.ndarray, List[np.ndarray]],
                       eps: float = 0.,
                       step: float = 5.
    ) -> float:
        '''
        Approximately evaluates the cost function by iteratively increasing the slack
        parameter until the computation is successful.

        Parameters
        ----
        A: np.ndarray
            State matrix A.
        B: np.ndarray or list[np.ndarray]
            Input-state matrix B or list of submatrices of B according to actuator schedule,
            where the index in the list corresponds to the time instant.
        eps: float, optional
            Slack to compute the cost function.
            If the first iteration is unsuccessful, the minimal subsequent value of eps is set to 1e-12.
        step:
            Multiplicative step that increases the slack after each unsuccessful computation.

        Returns
        ----
        cost: float,
            the value of the cost function.
        '''
        done = False
        while not done:
            try:
                cost = self.compute(A, B, eps)
                done = True
            except LinAlgError:
                eps = max(EPS, step * eps)

        return cost


    def log_det_cost(self, eps: float = 0.) -> float:
        '''
        Evaluates the function -log(det(W + eps I)).
        '''
        _, logabsdet = np.linalg.slogdet(self._W + eps * np.eye(len(self._W)))
        return -logabsdet


    def trace_cost(self) -> float:
        '''
        Evaluates the function 1/Tr(W).
        '''
        return 1/np.trace(self._W)
    

    def trace_inv_cost(self, eps: float = 0.) -> float:
        '''
        Evaluates the function Tr(W^(-1) + eps I).
        '''
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fxn()
            return np.trace(
                scipy.linalg.solve(
                    self._W + eps * np.eye(len(self._W)),
                    np.eye(len(self._W)),
                    lower=True,
                    assume_a='pos',
                    overwrite_b=True
                )
            )


    def lambda_min_cost(self, eps: float = 0.) -> float:
        '''
        Evaluates the function 1\lambda_min(W + eps I)
        '''
        singular_values = np.linalg.svd(self._W + eps * np.eye(len(self._W)), compute_uv=False)
        return 1/min(singular_values) if singular_values.all() else np.inf
    

    def update_gramian(self,
                A: Union[np.ndarray, List[np.ndarray]],
                B: Union[np.ndarray, List[np.ndarray]],
                h: Union[int, NoneType] = None
    ) -> np.ndarray:
        '''
        Computes the controllability Gramian.

        Parameters
        ----
        A: np.ndarray
            State matrix A.
        B: np.ndarray or list[np.ndarray]
            Input-state matrix B or list of submatrices of B according to actuator schedule,
            where the index in the list corresponds to the time instant.
        h: int
            Time horizon of the reachability matrix.
            If None, the attribute self.h is used.
        '''
        if h is None:
            h = self.h

        if isinstance(A, np.ndarray) and isinstance(B, np.ndarray):
            self._contr_mat = B
            self._W = np.matmul(B, B.T)
            for k in range(1, h):
                self._add_to_gramian(A, B, k)

        if isinstance(A, np.ndarray) and isinstance(B, list):
            self._contr_mat = B[-1]
            self._W = np.matmul(B[-1], B[-1].T)
            for k in range(1, min(h, len(B))):
                self._add_to_gramian(A, B[-1-k], k)

            for k in range(min(h, len(B)), h):
                self._add_to_gramian(A, B[0], k)
    

    def _add_to_gramian(self,
                       A: np.ndarray,
                       B: np.ndarray, 
                       k: int
    ) -> None:
        '''
        Adds the term A^k B to the reachability matrix and the associated term A^k B B^T (A^k)^T.
        '''
        phi_k = np.matmul(np.linalg.matrix_power(A, k), B)
        self._contr_mat = np.hstack([phi_k, self._contr_mat])
        self._W = self._W + np.matmul(phi_k, phi_k.T)