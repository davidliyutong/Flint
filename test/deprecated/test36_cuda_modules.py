import numpy as np
from core.Functions.cuda import *
from core.Layers import im2col
import time
from core.Initializer import xavieruniform

input = np.random.randn(3, 280, 280, 32).astype(np.float32)
input_gpu = cuda.to_device(input)
lbl = [0, 1, 0, 0, 0, 0, 0, 0, 0, 0]
kernel_shape = [3, 3, 32, 32]
col_gpu = cu_im2col(input_gpu, kernel_shape, from_host=False, to_host=False)
col = im2col(input, kernel_shape)

# ## efficiency
# start_t = time.time()
# for i in range(10000):
#     col = cu_conv_im2col(input_gpu , kernel_shape, from_host=False, to_host=False)
#
# print("[ duration ]: ", time.time() - start_t)
#
# start_t = time.time()
# for i in range(10000):
#     col = im2col(input, kernel_shape)
#
# print("[ duration ]: ", time.time() - start_t)

random_kernel = xavieruniform().init(shape=kernel_shape)
random_kernel_gpu = cuda.to_device(random_kernel)
random_kernel_reshaped = random_kernel.reshape(-1, random_kernel.shape[3])
random_kernel_reshaped_gpu = random_kernel_gpu.reshape(np.prod(random_kernel_gpu.shape[0:3]),
                                                       random_kernel_gpu.shape[3])

Z = np.matmul(col, random_kernel_reshaped)
Z_gpu = cu_matmul(col, random_kernel_reshaped, from_host=True, to_host=False)
Z_cpu = Z_gpu.copy_to_host()

## efficiency
start_t = time.time()
for i in range(1000):
    Z_gpu = cu_matmul(col_gpu, random_kernel_reshaped_gpu, from_host=False, to_host=False)

print("[ duration ]: ", time.time() - start_t)

start_t = time.time()
for i in range(1000):
    Z = np.matmul(col, random_kernel_reshaped)

print("[ duration ]: ", time.time() - start_t)

print("finish")
