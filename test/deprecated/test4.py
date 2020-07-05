import numpy as np
import numba.cuda as cuda
import numpy as np
from numba import cuda, float32
import math


# Controls threads per block and shared memory usage.
# The computation will be done on blocks of TPBxTPB elements.
@cuda.jit
def matmul(A, B, C):
    x, y = cuda.grid(2)
    if x >= C.shape[0] or y >= C.shape[1]:
        # Quit if (x, y) is outside of valid C boundary
        return
    tmp = 0.
    for i in range(A.shape[0]):
        tmp += A[x, i] * B[y, i]
    C[x, y] = tmp


weights = np.zeros(shape=(28 * 28 * 10, 10)).astype(np.float32)
weights = weights + 1
print('[ weights ]')
print(weights)

origin = np.zeros(shape=(10 * 28 * 28, 1)).astype(np.float32)
origin = origin + 1
print('[ origin ]')
print(origin)

output = np.zeros(shape=(10, 1)).astype(np.float32)

origin_gpu = cuda.to_device(origin)
weights_gpu = cuda.to_device(weights)
output_gpu = cuda.to_device(output)

matmul[(math.ceil(28 * 28 * 10 / 256), 10), 256](weights_gpu, origin_gpu, output_gpu)

output_gpu.copy_to_host(output)

print('[ result ]')
print(output)
