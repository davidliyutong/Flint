import numpy as np
from .pool import pool
from core.Functions.cuda import *


def _do_upsample(x, kernel_sz, last_input):
    assert len(x.shape) == 4
    batch_size = x.shape[0]
    upsampled_x = np.zeros(shape=(batch_size, x.shape[1] * kernel_sz, x.shape[2] * kernel_sz, x.shape[3]),
                           dtype=np.float32)

    for k in range(batch_size):
        for c in range(x.shape[3]):
            for j in range(x.shape[2]):
                for i in range(x.shape[1]):
                    idx = np.argmax(last_input[k, i:i + kernel_sz, j:j + kernel_sz, c])
                    idx_x = idx // kernel_sz
                    idx_y = idx % kernel_sz
                    upsampled_x[k, i * kernel_sz + idx_x, j * kernel_sz + idx_y, c] = x[k, i, j, c]

    return upsampled_x


def _do_max_pool(x, output_shape, kernel_sz):
    assert len(output_shape) == 3 and isinstance(kernel_sz, int)
    batch_sz = x.shape[0]
    output = np.zeros(shape=(batch_sz, *output_shape), dtype=np.float32)
    argmax = np.zeros(shape=(batch_sz, *output_shape), dtype=np.float32)
    for i in range(output_shape[0]):
        for j in range(output_shape[1]):
            pool = x[:, i:i+kernel_sz, j:j+kernel_sz, :]
            pool = pool.reshape((batch_sz, -1, output_shape[2]))

            _argmax = np.argmax(pool, axis=1)[:, np.newaxis,:]
            argmax[:, i, j, :] = _argmax.squeeze()

            _output = np.take_along_axis(pool, _argmax, axis=1).squeeze()
            output[:, i, j, :] = _output
    
    return output


class max_pool(pool):
    def __init__(self, input_shape=None, kernel_sz=1):
        super(max_pool, self).__init__(input_shape, kernel_sz)
        self._do_pool = _do_max_pool
        self._do_upsample = _do_upsample
        self._do_pool_cuda = cu_max_pool_ds
        self._do_upsample_cuda = cu_max_pool_us

    def compile(self, input_shape):
        super(max_pool, self).compile(input_shape)
        self.description['class'] = 'max_pool'
