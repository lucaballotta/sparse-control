import warnings
import numpy as np

from scipy.linalg import qr
from types import NoneType
from typing import *

EPS = 1e-10

def truncate_float(float_number, decimal_places):
    multiplier = 10 ** decimal_places
    return int(float_number * multiplier) / multiplier


def staircase(
        R: np.ndarray, 
        tol: float = EPS
) -> List[int]:
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


def independent_cols(
        A: Union[np.ndarray, NoneType],
        B: Union[np.ndarray, NoneType] = None,
        B_indep: bool = False
) -> Tuple[np.ndarray, List[int]]:
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
        return independent_cols(B, A)
    
    if B is None:
        R = qr(A, mode='r', check_finite=False)
        ind_col_idx = staircase(R[0])
        return A[:, ind_col_idx], ind_col_idx

    if B_indep:
        R = qr(np.hstack((B, A)), mode='r', check_finite=False)
        ind_col_idx = staircase(R[0])
        ind_col_idx_A = [idx - np.shape(B)[1] for idx in ind_col_idx if idx >= np.shape(B)[1]]
        return A[:, ind_col_idx_A], ind_col_idx_A
    
    else:
        R = qr(np.hstack((A, B)), mode='r', check_finite=False)
        ind_col_idx = staircase(R[0])
        ind_col_idx_A = [idx for idx in ind_col_idx if idx < np.shape(A)[1]]
        ind_col_idx_B = [idx - np.shape(A)[1] for idx in list(set(ind_col_idx) - set(ind_col_idx_A))]
        return np.hstack(
            (A[:, ind_col_idx_A],
                B[:, ind_col_idx_B])
        ), ind_col_idx
        

def left_kernel(
        A_k: np.ndarray,
        B: np.ndarray,
        A_prev: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, List[int]]:
    '''
    Finds the columns of B that span the subspace ker((A^(k+1))^T) - ker((A^k)^T)
        
    Params
    ----
    A_k: np.ndarray,
        matrix A^k
    B: np.ndarray,
        matrix B
    A_prev: no.ndarray
        matrix A^(k+1)

    Returns
    ---
    im_AB: np.ndarray
        matrix product A^k B
    im_K: np.ndarray
        matrix product [(I - A^(k+1) pinv(A^(k+1))) A^k]^T A^k B
    ch_ker: list[int]
        columns of B that span ker((A^(k+1))^T) - ker((A^k)^T)
    '''

    # image of B through A^k
    im_AB = np.matmul(A_k, B)
    im_AB[abs(im_AB) < EPS] = 0

    # subspace of left kernel of A^(k+1) orthogonal to left kernel of A^k
    K = np.matmul(
        np.eye(len(A_prev)) - np.matmul(A_prev, np.linalg.pinv(A_prev)),
        A_k
    )
    K[abs(K) < EPS] = 0

    # keep columns B_j of B s.t. A^k B_j is not orthogonal to K
    im_K = np.matmul(K.T, im_AB)
    im_K[abs(im_K) < EPS] = 0
    ch_ker = list(np.where(im_K.any(axis=0))[0])
    return im_AB, im_K, ch_ker


def fxn() -> None:
    warnings.warn("deprecated", DeprecationWarning)