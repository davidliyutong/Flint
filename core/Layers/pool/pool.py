import numpy as np
import math
from ..__init__ import layer
import cupy as cp
from core.Functions.cuda import *


def _do_padding(x, padding_shape, input_shape):
    assert len(input_shape) == 3 and len(padding_shape) == 3
    assert len(x.shape) == 4

    input_s_x = input_shape[0]
    input_s_y = input_shape[1]
    padding_s_x = padding_shape[0]
    padding_s_y = padding_shape[1]

    padded_x = np.pad(x,
                      ((0, 0), (0, padding_s_x - input_s_x), (0, padding_s_y - input_s_y), (0, 0)),
                      mode='constant',
                      constant_values=(0, 0))

    return padded_x


def _do_padding_cuda(x, padding_shape, input_shape):
    assert len(input_shape) == 3 and len(padding_shape) == 3
    assert len(x.shape) == 4

    input_s_x = input_shape[0]
    input_s_y = input_shape[1]
    padding_s_x = padding_shape[0]
    padding_s_y = padding_shape[1]

    padded_x = cp.pad(x,
                      ((0, 0), (0, padding_s_x - input_s_x), (0, padding_s_y - input_s_y), (0, 0)),
                      mode='constant',
                      constant_values=(0, 0))

    return padded_x


class pool(layer):
    def __init__(self, input_shape=None, kernel_sz=1):
        super(pool, self).__init__()

        self.input_shape = input_shape
        self.kernel_sz = kernel_sz

        if input_shape is not None:
            self.compile(self.input_shape)

        self._do_pool = None
        self._do_pool_cuda = None
        self._do_padding = _do_padding
        self._do_padding_cuda = _do_padding_cuda
        self._do_upsample = None
        self._do_upsample_cuda = None

    def compile(self, input_shape):
        assert input_shape is not None
        self.input_shape = input_shape

        # >>>> compile the model <<<<
        self.input_sz = self.input_shape[1]
        self.input_ch = self.input_shape[2]
        self.output_sz = math.ceil(self.input_sz / self.kernel_sz)
        self.output_c = self.input_ch
        self.output_shape = [self.output_sz, self.output_sz, self.output_c]
        self.padding_s = self.output_sz * self.kernel_sz
        self.padding_shape = (self.padding_s, self.padding_s, self.input_ch)

        self.is_compiled = True
        self.description = {'type': 'layer',
                            'class': 'pool',
                            'trainable': False,
                            'input_shape': self.input_shape}

    def forward(self, x):
        assert self.is_compiled
        self.last_input = x
        if not self.input_sz == self.padding_s:
            x_padded = self._do_padding(x, self.padding_shape, self.input_shape)
            self.output = self._do_pool(x_padded, self.output_shape, self.kernel_sz)
        else:
            self.output = self._do_pool(x, self.output_shape, self.kernel_sz)
        return self.output

    def forward_cuda(self, x):
        assert self.is_compiled
        self.last_input_cuda = x
        if not self.input_sz == self.padding_s:
            x_padded_cuda = self._do_padding_cuda(x, self.padding_shape, self.input_shape)
            self.output_cuda = self._do_pool_cuda(x_padded_cuda, self.output_shape, self.kernel_sz)
        else:
            self.output_cuda = self._do_pool_cuda(x, self.output_shape, self.kernel_sz)
        return self.output_cuda

    def backward(self, grad):
        assert self.is_compiled
        if not self.input_sz == self.padding_s:
            self.grad_upsampled = self._do_upsample(grad, self.kernel_sz, self.last_input)
            return self.grad_upsampled[:, 0:self.input_shape[0], 0:self.input_shape[1], :]
        else:
            return self._do_upsample(grad, self.kernel_sz, self.last_input)


    def backward_cuda(self, grad):
        assert self.is_compiled
        if not self.input_sz == self.padding_s:
            self.grad_upsampled_cuda = self._do_upsample_cuda(grad, self.kernel_sz, self.last_input_cuda)
            return self.grad_upsampled_cuda[:, 0:self.input_shape[0], 0:self.input_shape[1], :]
        else:
            return self._do_upsample_cuda(grad, self.kernel_sz, self.last_input_cuda)
