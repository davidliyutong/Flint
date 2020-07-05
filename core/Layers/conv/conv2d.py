import numpy as np
import math
from ..__init__ import layer
from core.Initializer import *
from core.Functions.cuda import cu_im2col
from core.Functions.cpu import im2col
import cupy as cp


class conv2d(layer):
    def __init__(self,
                 output_ch,
                 input_shape=None,
                 kernel_sz=3,
                 padding=0,
                 w_init=xavieruniform(),
                 b_init=zeros()):
        super(conv2d, self).__init__()

        self.input_shape = input_shape  # [n,m,in_c]
        self.kernel_shape = [kernel_sz, kernel_sz, None, output_ch]  # [k_h, k_w, in_c, out_c]
        self.kernel_sz = kernel_sz  # assert k_h == k_w  assume the kernels are spares
        self.output_ch = output_ch  # out_c output channel
        self.padding = padding  # padding size

        self.initializers = {"w": w_init, "b": b_init}  # initialize parameters
        self.shapes = {"w": [self.kernel_sz, self.kernel_sz, None, self.output_ch],
                       "b": [self.output_ch]}  # shape of parameters
        self.param_names = {"w", "b"}  # name of parameters
        self.params = {name: None for name in self.param_names}  # parameters

        if self.input_shape is not None:
            self.compile(self.input_shape)  # compile this layer

    def restore(self, state_dict: dict):
        # restore the parameters from saved models
        for name in self.param_names:
            assert self.shapes[name] == list(state_dict[name].shape)
            self.params[name] = state_dict[name]

    def _init_parames(self):
        # initialize the parameters using the defined initializer
        for p in self.param_names:
            self.params[p] = self.initializers[p](self.shapes[p])

    def compile(self, input_shape=None):
        assert input_shape is not None

        # calculate output shapes, etc,
        self.input_shape = input_shape
        self.input_ch = input_shape[2]
        self.kernel_shape[2] = self.input_shape[2]
        self.shapes["w"][2] = self.input_shape[2]
        self.output_sz = input_shape[0] + 1 - self.kernel_sz
        self.output_shape = [self.output_sz, self.output_sz, self.output_ch]

        # >>>> compile the model <<<<
        self._init_parames()
        # mark the layer as compiled
        self.is_compiled = True
        # Add description
        self.description = {'type': 'layer',
                            'class': 'conv2d',
                            'trainable': True,
                            'input_shape': self.input_shape,
                            'output_ch': self.output_ch,
                            'kernel_sz': self.kernel_sz,
                            'padding': self.padding
                            }
        # >>>> <<<<

    def forward(self, x):
        assert self.is_compiled
        self.last_input = x
        # reshape input
        x_col = im2col(x, self.kernel_shape)
        self.last_input_col = x_col
        # reshape kernel
        self.W = self.params["w"].reshape(-1, self.output_ch)
        # do the convolution
        Z = np.matmul(x_col, self.W)
        batch_sz = x.shape[0]
        Z = Z.reshape((batch_sz, Z.shape[0] // batch_sz, self.output_ch))
        Z = Z.reshape((batch_sz, self.output_sz, self.output_sz, self.output_ch))
        # add the bias
        Z += self.params["b"]

        return Z

    def forward_cuda(self, x):
        # the same routine
        assert self.is_compiled
        self.last_input_cuda = x
        X_cuda = cu_im2col(x, self.kernel_shape)
        self.last_input_col_cuda = X_cuda
        self.W_cuda = self.params_cuda["w"].reshape(np.prod(self.kernel_shape[0:3]), self.output_ch)
        Z_cuda = cp.matmul(X_cuda, self.W_cuda)
        batch_sz = x.shape[0]
        Z_cuda = Z_cuda.reshape((batch_sz, Z_cuda.shape[0] // batch_sz, self.output_ch))
        Z_cuda = Z_cuda.reshape((batch_sz, self.output_sz, self.output_sz, self.output_ch))
        Z_cuda += self.params_cuda["b"]

        return Z_cuda

    def backward(self, grad):
        assert self.is_compiled
        # reshape input gradient
        batch_sz = grad.shape[0]
        flat_grad = grad.reshape((-1, self.output_ch))

        # calcutate the dL / dW
        d_W = np.matmul(np.transpose(self.last_input_col), flat_grad)
        self.grads["w"] = d_W.reshape(self.kernel_shape)
        # calculate the dL / dB
        self.grads["b"] = np.sum(flat_grad, axis=0)

        # calculate the dL / dInput
        d_X = np.matmul(grad, np.transpose(self.W))
        d_in = np.zeros(shape=(batch_sz, *self.input_shape))

        # reshape the gradient of input
        for i, r in enumerate(range(0, self.output_sz)):
            for j, c in enumerate(range(0, self.output_sz)):
                patch = d_X[:, i, j, :]
                patch = patch.reshape((batch_sz, self.kernel_sz, self.kernel_sz, self.input_ch))
                d_in[:, r:r + self.kernel_sz, c:c + self.kernel_sz, :] += patch

        return d_in

    def backward_cuda(self, grad):
        # the same routine
        assert self.is_compiled

        batch_sz = grad.shape[0]
        flat_grad_cuda = grad.reshape((-1, self.output_ch))

        d_W_cuda = cp.matmul(cp.transpose(self.last_input_col_cuda), flat_grad_cuda)
        self.grads_cuda["w"] = d_W_cuda.reshape(self.kernel_shape)
        self.grads_cuda["b"] = cp.sum(flat_grad_cuda, axis=0)

        d_X_cuda = cp.matmul(grad, cp.transpose(self.W_cuda))
        d_in_cuda = cp.zeros(shape=(batch_sz, *self.input_shape), dtype=np.float32)

        # if this part could be re-written, it would be better
        for i, r in enumerate(range(0, self.output_sz)):
            for j, c in enumerate(range(0, self.output_sz)):
                # patch = d_X_cuda[:, i, j, :]
                # patch = patch.reshape((batch_sz, self.kernel_sz, self.kernel_sz, self.input_ch))
                # d_in_cuda[:, r:r + self.kernel_sz, c:c + self.kernel_sz, :] += patch
                d_in_cuda[:, r:r + self.kernel_sz, c:c + self.kernel_sz, :] += d_X_cuda[:, i, j, :].reshape(
                    (batch_sz, self.kernel_sz, self.kernel_sz, self.input_ch))

        return d_in_cuda
