from numba import cuda, float32, int32
import math
import numpy as np
import time

from test25_cu_lib import *

if __name__ == '__main__':
    A = np.random.randn(32, 64).astype(np.float32)
    B = np.random.randn(64, 32).astype(np.float32)
    C = np.zeros(shape=(32, 32)).astype(np.float32)

    A_gpu = cuda.to_device(A)
    B_gpu = cuda.to_device(B)
    C_gpu = cuda.to_device(C)

    # test speed
    # start_t = time.time()
    # for i in range(100):
    #     C = np.matmul(A, B)
    # print('[ cpu ]: ', time.time() - start_t)
    #
    # grid_dim, block_dim = auto_detect(C.shape)
    # start_t = time.time()
    # for i in range(100):
    #     _matmul[grid_dim, block_dim](A_gpu, B_gpu, C_gpu)
    # print('[ gpu ]: ', time.time() - start_t)
    #
    # start_t = time.time()
    # for i in range(100):
    #     cu_square_matrix_mul[(100,100), (32,32)](A_gpu, B_gpu, C_gpu)
    # print('[ gpu ]: ', time.time() - start_t)

    # # test matmul
    # C = np.matmul(A, B)
    # grid_dim, block_dim = auto_detect(C.shape)
    # C_gpu = cu_matmul(A_gpu, B_gpu)
    # print(C)
    # C_cpu = C_gpu.copy_to_host()
    # print(C_cpu)
    # print(np.sum(np.abs(C - C_cpu) > 0.001))

    # # test sum
    # A = np.random.randn(100, 200).astype(np.float32)
    # B = np.random.randn(200, 100).astype(np.float32)
    # C = np.zeros(shape=(100, 100)).astype(np.float32)
    #
    # A_gpu = cuda.to_device(A)
    # B_gpu = cuda.to_device(B)
    # C_gpu = cuda.to_device(C)
    #
    # D = np.sum(B, axis=0)
    # # D_gpu = cuda.device_array(shape=(200, 1), dtype=np.float32)
    # # column = np.ones(shape=(100, 1)).astype(np.float32)
    # # column_gpu = cuda.to_device(column)
    # # grid_dim, block_dim = auto_detect(D.shape)
    # # cu_matmul[grid_dim, block_dim](B, column_gpu, D_gpu)
    # D_gpu = cu_matsum_2d_0(B_gpu)
    # D_cpu = D_gpu.copy_to_host()
    # print(D)
    # print(D_cpu)
    # print(np.sum(np.abs(D - D_cpu) > 0.001))

    # B = np.random.randn(64, 1).astype(np.float32)
    # B_exp = np.exp(B)
    # B_exp_gpu = cu_matexp_2d(B, from_host=True, to_host=True)
    # print(B_exp)
    # print(B_exp_gpu)
    # print(np.sum(np.abs(B_exp - B_exp_gpu) > 0.001))

    # # mat_pow
    # B_pow = np.power(B,2)
    # B_pow_gpu = cu_matpow_2d(B, 2, from_host=True, to_host=True)
    # print(np.sum(np.abs(B_pow - B_pow_gpu) > 0.001))

    # # broadcast
    # D = np.random.randn(32, 1).astype(np.float32)
    # E = A / D
    # E_gpu = cu_mat_broadcast_multiply_2d_1(A, D, -1, from_host=True, to_host=True)
    # print(np.sum(np.abs(E - E_gpu) > 0.001))

    # # multiply
    # D = np.random.randn(32, 64).astype(np.float32)
    # E = A / D
    # E_gpu = cu_matmultply_2d(A, D, -1, from_host=True, to_host=True)
    # print(np.sum(np.abs(E - E_gpu) > 0.001))

    # # add
    # D = np.random.randn(32, 64).astype(np.float32)
    # E = A - D
    # E_gpu = cu_matadd_2d(A, D, - 1, from_host=True, to_host=True)
    # print(np.sum(np.abs(E - E_gpu) > 0.001))

    # # scalar
    # D = np.random.randn(32, 64).astype(np.float32)
    # E = 2 * D -1
    # E_gpu = cu_mat_scalar_2d(D, 2, - 1, from_host=True, to_host=True)
    # print(np.sum(np.abs(E - E_gpu) > 0.001))

    # # scalar
    # D = np.random.randn(32, 64).astype(np.float32)
    # E = np.max(D)
    # print(E)
    # # print(np.sum(np.abs(E - E_gpu) > 0.001))

    # reshape
    D = np.random.randn(4, 4, 3, 2).astype(np.float32)
    D_gpu = cuda.to_device(D)
    D_shape_gpu = D_gpu.reshape(4, 4 * 3 * 2)
    D_shape_cpu = D_shape_gpu.copy_to_host()
    D_reshape_gpu = D_shape_gpu.reshape(4, 4, 3, 2)
    D_reshape_cpu = D_reshape_gpu.copy_to_host()
    print(D)
    print(D_shape_cpu)
    print(D_reshape_cpu)
    print(np.sum(np.abs(D - D_reshape_cpu) > 0.001))
