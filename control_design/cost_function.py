import numpy as np
import scipy
import warnings

from typing import *
from matplotlib.pylab import LinAlgError

from .utils import fxn

EPS = 1e-10
COST_FUNCTIONS = {'logdet', 'tr', 'tr-inv', 'lambda-min'}

class CostFunction:

    def __init__(self, h: int, cost_func: str) -> None:
        if cost_func not in COST_FUNCTIONS:
            raise NotImplementedError('Cost function not implemented')
        
        if h <= 0:
            raise ValueError('InputError: horizon h must be greater than zero')
        
        self.cost_func = cost_func
        self.h = h
        self._contr_mat = None
        self._W = None
            
        
    def get_gramian(self):
        return self._W
    

    def get_gramian_rank(self):
        return np.linalg.matrix_rank(self._W)
    

    def compute(self,
                A: Union[np.ndarray, list[np.ndarray]],
                B: Union[np.ndarray, list[np.ndarray]],
                eps: float = 0.
    ) -> float:
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
                       A: Union[np.ndarray, list[np.ndarray]],
                       B: Union[np.ndarray, list[np.ndarray]],
                       eps: float = 0.
    ) -> float:
        done = False
        while not done:
            try:
                cost_best = self.compute(A, B, eps)
                done = True
            except LinAlgError:
                eps = max(EPS, 5 * eps)

        return cost_best


    def log_det_cost(self, eps: float = 0.) -> float:
        _, logabsdet = np.linalg.slogdet(self._W + eps * np.eye(len(self._W)))
        return -logabsdet


    def trace_cost(self) -> float:
        return 1/np.trace(self._W)
    

    def trace_inv_cost(self, eps: float = 0.) -> float:
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
        singular_values = np.linalg.svd(self._W + eps * np.eye(len(self._W)), compute_uv=False)
        return 1/min(singular_values) if singular_values.all() else np.inf
    

    def update_gramian(self,
                A: Union[np.ndarray, list[np.ndarray]],
                B: Union[np.ndarray, list[np.ndarray]],
                h: int = None
    ) -> np.ndarray:
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
        phi_k = np.matmul(np.linalg.matrix_power(A, k), B)
        self._contr_mat = np.hstack([phi_k, self._contr_mat])
        self._W = self._W + np.matmul(phi_k, phi_k.T)