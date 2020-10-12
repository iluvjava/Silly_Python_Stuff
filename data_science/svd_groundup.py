"""
    Here we take the time to use Eigen value decomposition subroutine
    to carry out a singular value decomposition for a randomly
    generated matrix.

    * Generate random integer matrix.
    * Compute the Eigen decomposition for the hermitian product

"""

import numpy as np
np.set_printoptions(precision=3)

def svd_demo():
    eigh = np.linalg.eigh
    diag = np.diag
    M = np.random.random((2, 7))
    M = M * 10
    M = M.astype(int)
    MTM = M.T@M
    Lambdas, V = eigh(MTM)
    Lambdas = Lambdas.round(8)
    SigmaSqrtInv = np.array([1/E**0.5 if E != 0 else E for E in Lambdas])
    print("Initial Matrix")
    print(M)
    print("Hermitian Product")
    print(MTM)
    print(Lambdas.round(8))
    print(V)
    print("The V has to be unitary! ")
    print((V.T@V).round(8))
    print((V@V.T).round(8))
    U = M@V@diag(SigmaSqrtInv)
    print(U)
    print(U@diag(Lambdas**0.5)@V.T)


def svd_with_eigh(m):
    eigh = np.linalg.eigh; diag = np.diag
    Lambdas, V = eigh(m.T@m)
    Lambdas = Lambdas.round(10)
    SigmaSqrtInv = np.array([1 / E ** 0.5 if E != 0 else E for E in Lambdas])
    U = m@V@diag(SigmaSqrtInv)
    return U, diag(Lambdas**0.5), V


def svd_with_eigh_sorted(m):
    """
        SVD a matrix with eigen decomposition on m@m.T
    :param m:
        any matrix
    :return:
        The decomposed S, Sigma, V matrices.
    """
    eigh = np.linalg.eigh
    diag = np.diag
    Lambdas, V = eigh(m.T @ m)
    Lambdas = Lambdas.round(10)
    EigenTuple = []
    for I in range(len(Lambdas)):
        EigenTuple.append((Lambdas[I], V[:, I]))
    EigenTuple = sorted(EigenTuple, key=lambda x: x[0], reverse=True)
    Lambdas = sorted([L for L in Lambdas], reverse=True)
    V = np.array([T[1] for T in EigenTuple]).T
    SigmaSqrtInv = np.array([1 / E ** 0.5 if E != 0 else E for E in Lambdas])
    U = m @ V @ diag(SigmaSqrtInv)
    return U, diag(np.array(Lambdas)**0.5), V


def double_eigh_demo():
    eigh = np.linalg.eigh; diag = np.diag
    M = np.random.random((3, 10))
    Lambdas1, V = eigh(M.T@M)
    Lambdas2, U = eigh(M@M.T)
    print("2 of the list of lambdas are")
    print(Lambdas1)
    print(Lambdas2)
    print("Notice how, regardless of which decomposition we use, the non-zero eigen values " +
          "are the same. ")

def main():
    RandomMatrix = np.random.random((10, 10))
    U, Sigma, V = svd_with_eigh_sorted(RandomMatrix)

    print("U, Sigma, V")
    print(U)
    print(Sigma)
    print(V)
    print("U@Sigma@V^T")
    print(U@Sigma@V.T)
    print("Random Matrix")
    print(RandomMatrix)


if __name__ == "__main__":
    main()
