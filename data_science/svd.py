"""
    In this file, we try to use SVD for data science.
"""

import numpy as np


def svd_with_eigh_sorted(m):
    """
        SVD a matrix with eigen decomposition on m@m.T
        * Full Decomposition when matrix is fat
        * Partial when matrix is skinny
    :param m:
        any matrix
    :return:
        The decomposed S, Sigma, V matrices.
    """
    eigh = np.linalg.eigh
    diag = np.diag
    Lambdas, V = eigh(m.T @ m)
    Lambdas = Lambdas.round(16)
    EigenTuple = []
    for I in range(len(Lambdas)):
        EigenTuple.append((Lambdas[I], V[:, I]))
    EigenTuple = sorted(EigenTuple, key=lambda x: x[0], reverse=True)
    Lambdas = sorted([L for L in Lambdas], reverse=True)
    V = np.array([T[1] for T in EigenTuple]).T
    SigmaSqrtInv = np.array([1 / E ** 0.5 if E != 0 else E for E in Lambdas])
    U = m @ V @ diag(SigmaSqrtInv)
    return U, diag(np.array(Lambdas)**0.5), V
