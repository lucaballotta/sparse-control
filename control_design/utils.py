import warnings
import numpy as np

from scipy.linalg import qr

EPS = 1e-10

def truncate_float (float_number, decimal_places):
    multiplier = 10 ** decimal_places
    return int(float_number * multiplier) / multiplier


def staircase(R: np.ndarray, tol: float = EPS) -> list[int]:
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


def independent_cols(A: np.ndarray,
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
        

def left_kernel(A_curr, B_curr, A_prod):

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


def fxn():
    warnings.warn("deprecated", DeprecationWarning)