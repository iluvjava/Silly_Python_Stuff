# Must run this.
from numpy import identity, array, zeros, triu, set_printoptions
from numpy.linalg import norm
from math import isclose
eye = identity
arr = array
set_printoptions(precision=4)


__all__ = ["qr_factor"]

def qr_factor(R):
    """
        Performs a Householder Transformation on the given matrix A,
        full QR Decomposition.
    """
    R = R.copy().astype("float64")
    assert len(R.shape) == 2
    m, n = R.shape[0], R.shape[1]
    Q = eye(m)
    for K in range((n - 1) if n == m else n):
        z = R[K:m, [K]]
        v = zeros((z.shape[0], 1))
        NormZ = norm(z)
        if isclose(NormZ, 0): raise Exception("Rank Defecit")
        v[0, 0] = (1 if z[0] < 0 else -1)*NormZ
        v = v - z
        v = v/norm(v)
        J = list(range(n))
        R[K: m, :n] = R[K: m, J] - 2*(v@v.T)@R[K: m, J]
        J = list(range(m))
        Q[K: m, :m] = Q[K: m, J] - 2*(v@v.T)@Q[K: m, J]
    return Q.T, triu(R)


def main():
    def PathologicalInput1():
        A = arr([[0, 1], [1, 1]])
        print(A)
        Q, R = qr_factor(A)
        print("These are the factors: ")
        print(Q)
        print(R)
        print("Matrix Reconstructed: ")
        print(Q @ R)

    A = arr([[0, 1], [1, 0], [1, 0]])
    print(A)
    Q, R = qr_factor(A)
    print("These are the factors: ")
    print(Q)
    print(R)
    print("Matrix Reconstructed: ")
    print(Q @ R)

    pass


if __name__ == "__main__":
    main()
