import numpy as np
from .loss import loss
import cupy as cp


def softmax(x, t=1.0, axis=-1):
    x_tmp = x / t
    x_max = np.max(x_tmp, axis=axis, keepdims=True)
    exps = np.exp(x_tmp - x_max)
    return exps / np.sum(exps, axis=axis, keepdims=True)


def log_softmax(x, t=1.0, axis=-1):
    x_tmp = x / t
    x_max = np.max(x_tmp, axis=axis, keepdims=True)
    exps = np.exp(x_tmp - x_max)
    exp_sum = np.sum(exps, axis=axis, keepdims=True)
    return x_tmp - x_max - np.log(exp_sum)


def softmax_cuda(x, t=1.0, axis=-1):
    x_tmp = x / t
    x_max = cp.max(x_tmp, axis=axis, keepdims=True)
    exps = cp.exp(x_tmp - x_max)
    return exps / cp.sum(exps, axis=axis, keepdims=True)


def log_softmax_cuda(x, t=1.0, axis=-1):
    x_tmp = x / t
    x_max = cp.max(x_tmp, axis=axis, keepdims=True)
    exps = cp.exp(x_tmp - x_max)
    exp_sum = cp.sum(exps, axis=axis, keepdims=True)
    return x_tmp - x_max - cp.log(exp_sum)


class softmax_cross_entropy(loss):
    def __init__(self, T=1.0, weight=None):
        """
        :reference: https://www.mldawn.com/binary-classification-from-scratch-using-numpy/
        """
        super(softmax_cross_entropy, self).__init__()

        weight = np.asarray(weight) if weight is not None else weight
        weight_gpu = cp.array(weight) if weight is not None else weight

        self._weight = weight
        self._weight_gpu = weight_gpu
        self._T = T

    def loss(self, x, labels):
        batch_sz = x.shape[0]
        nll = -(log_softmax(x, t=self._T, axis=1) * labels).sum(axis=1)

        if self._weight is not None:
            nll *= self._weight[labels]
        return np.sum(nll) / batch_sz

    def loss_cuda(self, x, labels):
        batch_sz = x.shape[0]
        nll = -(log_softmax_cuda(x, t=self._T, axis=1) * labels).sum(axis=1)

        if self._weight_gpu is not None:
            nll *= self._weight_gpu[labels]
        return cp.sum(nll) / batch_sz

    def grad(self, x, labels):
        batch_sz = x.shape[0]
        self.grads_cuda["i"] = (softmax(x, t=self._T) - labels) / batch_sz
        return self.grads_cuda["i"]

    def grad_cuda(self, x, labels):
        batch_sz = x.shape[0]
        self.grads_cuda["i"] = (softmax_cuda(x, t=self._T) - labels) / batch_sz
        return self.grads_cuda["i"]
