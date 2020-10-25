# Must run this.
import numpy as np
import math as math
sign = np.sign
eye = np.identity
sqrt = math.sqrt
norm = np.linalg.norm
arr = np.array
vstack = np.vstack
rand = np.random.rand
zeros = np.zeros
triu = np.triu
tril = np.tril
np.set_printoptions(precision=4)


def qr_factor(A):
    """
        Performs a Householder Transformation on the given matrix A.
    """
    A = A.copy()
    assert len(A.shape) == 2
    m, n = A.shape[0], A.shape[1]
    Q = eye(m)
    for K in range(n - 1):
        z = A[K:m, [K]]
        v = zeros((z.shape[0], 1))
        v[0, 0] = -sign(z[0])*norm(z)
        v = v - z
        v = v/norm(v)
        J = list(range(K, n))
        A[K: m, J] = A[K: m, J] - 2*(v@v.T@A[K: m, J])
        J = list(range(m))
        Q[K: m, J] = Q[K: m, J] - 2*(v@v.T@Q[K: m, J])
    return Q.T,  A


def main():
    M = rand(3, 3) * (10)
    M = M.round(0)
    print("This is the original matrix: ")
    print(M)
    Q, R = qr_factor(M)
    print("This is its factor, Q, R")
    print(Q)
    print(R)
    print("This is the reconstruction of the matrix: ")
    print(Q @ R)
    print("This is Q Q transpose")
    print(Q @ Q.T)

    M = rand(1000, 100) * (10)
    M = M.round(0)
    Q, R = qr_factor(M)
    print(norm((Q @ R - M)))
    print(norm(Q @ Q.T - eye(Q.shape[0])))

    M = arr([[0, 1, 1], [1, 1, 1], [1, 1, 1]])
    pass


if __name__ == "__main__":
    main()
