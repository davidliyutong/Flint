import numpy as np
import scipy.linalg.blas as blas
import numba.cuda as cuda
import pyculib.blas as cublas

A = np.random.randn(3, 3)
B = np.random.randn(3, 3)

C = blas.sgemm(1.0, A, B)
print(C)

A_d = cuda.to_device(A)
B_d = cuda.to_device(B)

C_d = cublas.gemm("N", "N", 1.0, A_d, B_d)
print(C_d)
