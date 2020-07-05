import numpy as np
import math
from core.Layers import layer
import cupy as cp


class activation(layer):
    def __init__(self, input_shape=None):
        super(activation, self).__init__()
        self.input_shape = input_shape

        if input_shape is not None:
            self.compile(self.input_shape)

        self.description = {'type': 'layer',
                            'class': 'activ',
                            'trainable': False,
                            'input_shape': self.input_shape}

    def compile(self, input_shape):
        assert input_shape is not None
        self.input_shape = input_shape
        # >>>> compile the model <<<<
        self.output_shape = self.input_shape
        self.is_compiled = True
        self.description['input_shape'] = self.input_shape
        # >>>> <<<<


class tanh(activation):
    def __init__(self, input_shape=None):
        super(tanh, self).__init__(input_shape)
        self.description['class'] = 'tanh'

    def forward(self, x):
        self.last_input = x
        return np.tanh(x)

    def forward_cuda(self, x):
        self.last_input_cuda = x
        return cp.tanh(x)

    def backward(self, grad):
        curr_gradient = 1 - np.power(self.output, 2)
        return np.multiply(curr_gradient, grad)

    def backward_cuda(self, grad):
        curr_gradient_cuda = 1 - cp.power(self.output_cuda, 2)
        return cp.multiply(curr_gradient_cuda, grad)


class relu(activation):
    def __init__(self, input_shape=None):
        super(relu, self).__init__(input_shape)
        self.description['class'] = 'relu'

    def forward(self, x):
        self.last_input = x
        return np.maximum(x, 0)

    def forward_cuda(self, x):
        self.last_input_cuda = x
        return cp.maximum(x, 0)

    def backward(self, grad):
        return np.multiply(self.last_input > 0, grad)

    def backward_cuda(self, grad):
        return cp.multiply(self.last_input_cuda > 0, grad)


class sigmoid(activation):
    def __init__(self, input_shape=None):
        super(sigmoid, self).__init__(input_shape)
        self.description['class'] = 'tanh'

    def forward(self, x):
        self.last_input = x
        return 1 / (1 + np.exp(-x))

    def forward_cuda(self, x):
        self.last_input_cuda = x
        return 1 / (1 + cp.exp(-x))

    def backward(self, grad):
        curr_gradient = self.output * (1 - self.output)
        return np.multiply(curr_gradient, grad)

    def back_cuda(self, grad):
        curr_gradient_cuda = self.output_cuda * (1 - self.output_cuda)
        return cp.multiply(curr_gradient_cuda, grad)
