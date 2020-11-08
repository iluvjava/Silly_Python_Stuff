from numpy import identity, array, triu,set_printoptions, argmax
from numpy.random import rand
from math import isclose
eye = identity
arr = array
set_printoptions(precision=4)

__all__  = ["lu_decompose"]

def lu_decompose(A):
    """
        Function will performs a LU decomposition on the given matrix
    :param A:
        A invertible matrix.
    :return:
        P, L, U such that PA = LU
    """
    M, N = A.shape
    if M != N:
        raise Exception("Matrix must be squared. ")
    U = A.copy().astype("float64"); L = eye(M); P = eye(M)
    for K in range(N - 1):
        I = argmax(abs(U[K:, K])) + K
        assert not isclose(U[I, K], 0), "Hardly Invertible Matrix"
        U[[K], K:], U[[I], K:] = U[[I], K:], U[[K], K:]
        if K >= 1: L[[K], :K], L[[I], :K] = L[[I], :K], L[[K], :K]
        P[[K], :], P[[I], :] = P[[I], :], P[[K], :]
        # Cancellations
        L[K + 1:, K] = U[K + 1:, K]/U[K, K]
        for J in range(K + 1, M):
            U[J, K:] = U[J, K:] - L[J, K]*U[K, K:]
    return P, L, triu(U)


def main():
    #M = arr([[2, 3, 4], [8, 7, 8], [5, 6, 7]])
    M = (rand(5, 5)*10).astype("int")
    print("The random matrix is: ")
    print(M)
    P, L, U = lu_decompose(M)
    print(P)
    print(L)
    print(U)
    print("P.TLU is: ")
    print(P.T@L@U)
    pass


if __name__ == "__main__":
    main()
    
