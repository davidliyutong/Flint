from .loss import loss
import numpy as np
import cupy as cp


class mse(loss):
    def __int__(self):
        super(mse, self).__init__()

    def loss(self, x, labels):
        batch_sz = x.shape[0]
        return 0.5 * np.sum((x - labels) ** 2) / batch_sz

    def grad(self, x, labels):
        batch_sz = x.shape[0]
        return (x - labels) / batch_sz

    def loss_cuda(self, x, labels):
        batch_sz = x.shape[0]
        return 0.5 * cp.sum((x - labels) ** 2) / batch_sz

    def grad_cuda(self, x, labels):
        batch_sz = x.shape[0]
        return (x - labels) / batch_sz
