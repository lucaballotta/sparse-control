import numpy as np

from typing import *

class CostFunction:

    def __init__(self, h: int, cost_func: str) -> None:
        self.h = h
        self.contr_mat = None
        self.W = None
        if cost_func == 'logdet':
            self.compute = self.log_det_cost
        elif cost_func == 'tr':
            self.compute = self.trace_cost
        elif cost_func == 'inv-tr':
            self.compute = self.inv_trace_cost
        elif cost_func == 'lam-min':
            self.compute = self.lam_min_cost
        else:
            raise NotImplementedError('Cost function not implemented')


    def log_det_cost(self, 
                     A: Union[np.ndarray, list[np.ndarray]],
                     B: Union[np.ndarray, list[np.ndarray]],
                     eps: float = 0.
    ) -> float:
        self.gramian(A, B)
        _, logabsdet = np.linalg.slogdet(self.W)
        return -logabsdet


    def trace_cost(self,
                   A: Union[np.ndarray, list[np.ndarray]],
                   B: Union[np.ndarray, list[np.ndarray]],
                   eps: float = 0.
    ) -> float:
        self.gramian(A, B)
        return 1/np.trace(self.W)
    

    def inv_trace_cost(self, 
                       A: Union[np.ndarray, list[np.ndarray]],
                       B: Union[np.ndarray, list[np.ndarray]],
                       eps: float = 0.
    ) -> float:
        self.gramian(A, B)
        return np.trace(np.linalg.inv(self.W))

    def lam_min_cost(self,
                     A: Union[np.ndarray, list[np.ndarray]],
                     B: Union[np.ndarray, list[np.ndarray]],
                     eps: float = 0.
    ) -> float:
        self.gramian(A, B)
        eig_W = np.real(np.linalg.eigvals(self.W))
        return 1/min(eig_W)
    

    def gramian(self,
                A: Union[np.ndarray, list[np.ndarray]],
                B: Union[np.ndarray, list[np.ndarray]],
                h: int = None
    ) -> np.ndarray:
        if h is None:
            h = self.h

        if isinstance(A, np.ndarray) and isinstance(B, np.ndarray):
            self.contr_mat = B
            self.W = np.matmul(B, B.T)
            for k in range(1, h):
                self.add_to_gramian(A, B, k)

        if isinstance(A, np.ndarray) and isinstance(B, list):
            self.contr_mat = B[-1]
            self.W = np.matmul(B[-1], B[-1].T)
            for k in range(1, min(h, len(B))):
                self.add_to_gramian(A, B[-1-k], k)

            for k in range(min(h, len(B)), h):
                self.add_to_gramian(A, B[0], k)
    

    def add_to_gramian(self,
                       A: np.ndarray,
                       B: np.ndarray, 
                       k: int
    ) -> None:
        phi_k = np.matmul(np.linalg.matrix_power(A, k), B)
        self.contr_mat = np.hstack([phi_k, self.contr_mat])
        self.W = self.W + np.matmul(phi_k, phi_k.T)