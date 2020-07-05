import math
from .pool import pool
import numpy as np
from core.Functions.cuda import *


def _do_upsample(x, kernel_sz, *args):
    assert len(x.shape) == 4
    batch_size = x.shape[0]
    upsampled_x = np.empty(shape=(batch_size, x.shape[1] * kernel_sz, x.shape[2] * kernel_sz, x.shape[3]),
                           dtype=np.float32)
    span = (kernel_sz * kernel_sz)
    for k in range(batch_size):
        for c in range(x.shape[3]):
            for j in range(x.shape[2]):
                for i in range(x.shape[1]):
                    upsampled_x[k, i * kernel_sz:(i + 1) * kernel_sz, j * kernel_sz:(j + 1) * kernel_sz, c] = x[
                        k, i, j, c]

    return upsampled_x


def _do_avg_pool(x, output_shape, kernel_sz):
    assert len(output_shape) == 3 and isinstance(kernel_sz, int)
    batch_size = x.shape[0]
    output = np.empty(shape=(batch_size, *output_shape), dtype=np.float32)
    span = (kernel_sz * kernel_sz)

    for k in range(batch_size):
        for c in range(output_shape[3]):
            for j in range(output_shape[21]):
                for i in range(output_shape[1]):
                    output[i, j, c, k] = np.sum(x[k, i:i + kernel_sz, j:j + kernel_sz, c]) / span
    return output


class avg_pool(pool):
    def __init__(self, input_dim=None, kernel_sz=1):
        super(avg_pool, self).__init__(input_dim, kernel_sz)
        self._do_pool = _do_avg_pool
        self._do_upsample = _do_upsample
        self._do_pool_cuda = cu_avg_pool_ds
        self._do_upsample_cuda = cu_avg_pool_us

    def compile(self, input_dim, batch_size=1):
        super(avg_pool, self).compile(input_dim)
        self.description['class'] = 'avg_pool'
