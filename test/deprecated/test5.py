from numba import cuda
from numba import *
import numpy as np
import math
from timeit import default_timer as time

m = 100000
n = 100


@cuda.jit('void(f4[:,:], f4[:], f4[:])')
def cu_matrix_vector(A, b, c):
    row = cuda.grid(1)
    if (row < m):
        sum = 0

        for i in range(n):
            sum += A[row, i] * b[i]

        c[row] = sum


A = np.array(np.random.random((m, n)), dtype=np.float32)
B = np.array(np.random.random(m), dtype=np.float32)
C = np.empty_like(B)

s = time()
dA = cuda.to_device(A)
dB = cuda.to_device(B)
dC = cuda.to_device(C)

cu_matrix_vector[math.ceil((m + 511) / 512), 512](dA, dB, dC)

dC.to_host()

print(C)

e = time()
tcuda = e - s
print(tcuda)
