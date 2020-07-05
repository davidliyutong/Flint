import numpy as np
from core.Functions.cuda import *
from core.Layers import im2col
import time
from core.Initializer import xavieruniform
import cupy as cp

input = np.random.randn(3, 28, 28, 32).astype(np.float32)
input_gpu = cp.array(input)
lbl = [0, 1, 0, 0, 0, 0, 0, 0, 0, 0]
kernel_shape = [3, 3, 32, 32]

# col = im2col(input, kernel_shape)
# random_kernel = xavieruniform().init(shape=kernel_shape)
# random_kernel_reshaped = random_kernel.reshape(-1, random_kernel.shape[3])
# Z = np.matmul(col, random_kernel_reshaped)
#
# col_gpu = cp.array(col)
# random_kernel_reshaped_gpu = cp.array(random_kernel_reshaped)
# Z_gpu = cp.matmul(col_gpu, random_kernel_reshaped_gpu)
# Z_cpu = cp.asnumpy(Z_gpu)

# print("[ error ]", np.sum(np.abs(Z_cpu - Z) > 0.0001))

# add_kernel = cp.RawKernel(r'''
#  extern "C" __global__
#  void my_add(const float* x1, const float* x2, float* y) {
#      int tid = blockDim.x * blockIdx.x + threadIdx.x;
#      y[tid] = x1[tid] + x2[tid];
#  }
#  ''', 'my_add')
#
# x1 = cp.arange(25, dtype=cp.float32).reshape(5, 5)
# x2 = cp.arange(25, dtype=cp.float32).reshape(5, 5)
# y = cp.zeros((5, 5), dtype=cp.float32)
# add_kernel((5,), (5,), (x1, x2, y))  # grid, block and arguments
# print(y)

# ## efficiency
# start_t = time.time()
# for i in range(1000):
#     Z_gpu = cp.matmul(col_gpu, random_kernel_reshaped_gpu)
#
# print("[ duration ]: ", time.time() - start_t)
#
# start_t = time.time()
# for i in range(1000):
#     Z = np.matmul(col, random_kernel_reshaped)
#
# print("[ duration ]: ", time.time() - start_t)
#

# ## efficiency
# start_t = time.time()
# for i in range(1000):
#     col = im2col(input, kernel_shape)
#
# print("[ cpu duration ]: ", time.time() - start_t)
#
# start_t = time.time()
# for i in range(1000):
#     test = cu_conv_im2col(input_gpu, kernel_shape, from_host=False, to_host=False)
#
# print("[ gpu duration ]: ", time.time() - start_t)
#
#
# print("[ error ]", np.sum(np.abs(col-cp.asnumpy(test)) > 0.0001))

# input_padded = cp.pad(input_gpu,((0,0),(0,1),(0,1),(0,0)),mode='constant',constant_values=(4, 6))
# input_padded = input_padded.reshape(-1, input_padded.shape[3])
# input_padded = cp.asnumpy(input_padded)

kernel_sz = 5
batch_sz = 32
input_ch = 6
output_sz = 8
d_X = np.random.randn(32, 8, 8, 150)
d_in_cpu = np.zeros(shape=(32, 12, 12, 6), dtype=np.float32)
d_X_gpu = cp.array(d_X)
d_in_gpu = cp.array(d_in_cpu)

# for i, r in enumerate(range(0, output_sz)):
#     for j, c in enumerate(range(0, output_sz)):
#         patch = d_X[:, i, j, :]
#         patch = patch.reshape((batch_sz, kernel_sz, kernel_sz, input_ch))
#         d_in_cpu[:, r:r + kernel_sz, c:c + kernel_sz, :] += patch
#
# print(d_in_cpu)
#
# for i, r in enumerate(range(0, output_sz)):
#     for j, c in enumerate(range(0, output_sz)):
#         patch = d_X_gpu[:, i, j, :]
#         patch = patch.reshape((batch_sz, kernel_sz, kernel_sz, input_ch))
#         d_in_gpu[:, r:r + kernel_sz, c:c + kernel_sz, :] += patch
#
# print(d_in_gpu)
#
# print("[ error ]", np.sum(np.abs(d_in_cpu - cp.asnumpy(d_in_gpu)) > 0.0001))
# print("finish")

# start_t = time.time()
# for k in range(1000):
#     for i, r in enumerate(range(0, output_sz)):
#         for j, c in enumerate(range(0, output_sz)):
#             patch = d_X[:, i, j, :]
#             patch = patch.reshape((batch_sz, kernel_sz, kernel_sz, input_ch))
#             d_in_cpu[:, r:r + kernel_sz, c:c + kernel_sz, :] += patch
# print("[ duration ]: ", time.time() - start_t)
#
# start_t = time.time()
# for k in range(1000):
#     for i, r in enumerate(range(0, output_sz)):
#         for j, c in enumerate(range(0, output_sz)):
#             d_in_gpu[:, r:r + kernel_sz, c:c + kernel_sz, :] += d_X_gpu[:, i, j, :].reshape((batch_sz, kernel_sz, kernel_sz, input_ch))
# print("[ duration ]: ", time.time() - start_t)

# x = np.random.randn(32,28,28,32)
# x_gpu = cp.array(x)
# out = np.maximum(x, 0)
# out_gpu = np.maximum(x, 0)
# print("[ error ]", np.sum(np.abs(out - cp.asnumpy(out_gpu)) > 0.0001))
# print("finish")
#
# start_t = time.time()
# for k in range(1000):
#     out = np.maximum(x, 0)
# print("[ duration ]: ", time.time() - start_t)
#
# start_t = time.time()
# for k in range(1000):
#     out_gpu = cp.maximum(x_gpu, 0)
# print("[ duration ]: ", time.time() - start_t)

x = np.ones(shape=(32, 26, 26, 3)).astype(np.float32)
x = np.random.randn(32, 25, 25, 6).astype(np.float32)
x_gpu = cp.array(x)

x_max_pool_gpu = cu_max_pool_ds(x_gpu, (5, 5, 6), 5)
x_max_pool_cpu = cp.asnumpy(x_max_pool_gpu)

x_max_pool_us = cu_max_pool_us(x_max_pool_gpu, 5, x_gpu)

# x_avg_pool_gpu = cu_avg_pool_ds(x_gpu, (5,5,6), 5)
# x_avg_pool_cpu = cp.asnumpy(x_avg_pool_gpu)
#
# x_avg_pool_us = cu_avg_pool_us(x_avg_pool_gpu, 5, x_gpu)

print("finish")
