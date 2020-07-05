import numpy as np
import math
from .layer import layer


class flatten(layer):
    def __init__(self, input_shape=None):
        super(flatten, self).__init__()
        """
        :param input_dim: The dimension of input
        """
        self.input_shape = input_shape
        if self.input_shape is not None:
            self.compile(input_shape)

    def compile(self, input_shape):
        # >>>> compile the model <<<<
        self.input_shape = input_shape
        self.output_shape = int(np.prod(self.input_shape))
        self.is_compiled = True
        self.description = {'type': 'layer',
                            'class': 'flatten',
                            'trainable': False,
                            'input_shape': self.input_shape
                            }
        # >>>> <<<<

    def forward(self, x):
        assert self.is_compiled
        self.batch_sz = x.shape[0]
        return x.reshape(x.shape[0], -1)

    def backward(self, grad):
        assert self.is_compiled
        return grad.reshape(self.batch_sz, *self.input_shape)

    def forward_cuda(self, x):
        assert self.is_compiled
        self.batch_sz = x.shape[0]
        return x.reshape(x.shape[0], np.prod(x.shape[1:]))

    def backward_cuda(self, grad):
        assert self.is_compiled
        return grad.reshape(self.batch_sz, *self.input_shape)
