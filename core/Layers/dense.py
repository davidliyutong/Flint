import numpy as np
from .layer import layer
from core.Initializer import *
import cupy as cp


class dense(layer):
    def __init__(self, output_shape, input_shape=None, w_init=xavieruniform(), b_init=zeros()):
        super(dense, self).__init__()

        # layer initialize
        self.initializers = {"w": w_init, "b": b_init}
        self.shapes = {"w": [None, output_shape], "b": [output_shape]}
        self.param_names = ("w", "b")
        self.params = {name: None for name in self.param_names}

        assert isinstance(output_shape, int)
        self.output_shape = output_shape
        self.input_shape = input_shape

        if self.input_shape is not None:
            self.compile(self.input_shape)

    def restore(self, state_dict: dict):
        # restore parameters
        for name in self.param_names:
            assert self.shapes[name] == list(state_dict[name].shape)
            self.params[name] = state_dict[name]

    def _init_parames(self):
        for p in self.param_names:
            self.params[p] = self.initializers[p](self.shapes[p])

    def compile(self, input_shape):
        assert input_shape is not None
        self.input_shape = input_shape
        self.shapes["w"][0] = input_shape

        # >>>> compile the model <<<<
        self._init_parames()
        self.is_compiled = True
        self.description = {'type': 'layer',
                            'class': 'dense',
                            'trainable': True,
                            'input_shape': self.input_shape,
                            'output_features': self.output_shape}
        # >>>> <<<<

    def forward(self, x):
        assert self.is_compiled
        self.last_input = x
        return np.matmul(x, self.params["w"]) + self.params["b"]

    def forward_cuda(self, x):
        # same routine
        assert self.is_compiled
        self.last_input_cuda = x
        return cp.matmul(x, self.params_cuda["w"]) + self.params_cuda["b"]

    def backward(self, grad):
        assert self.is_compiled
        self.grads["b"] = np.sum(grad, axis=0)
        self.grads["w"] = np.matmul(np.transpose(self.last_input), grad)
        return np.matmul(grad, np.transpose(self.params["w"]))

    def backward_cuda(self, grad):
        assert self.is_compiled
        self.grads_cuda["b"] = cp.sum(grad, axis=0)
        self.grads_cuda["w"] = cp.matmul(cp.transpose(self.last_input_cuda), grad)
        return cp.matmul(grad, cp.transpose(self.params_cuda["w"]))
