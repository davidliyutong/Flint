from .optimizer import optimizer
import time
import numpy as np
import cupy as cp


class adam(optimizer):
    def __init__(self, lr=1e-4, beta1=0.9, beta2=0.999, eps=1e-8, weight_decay=0.0):
        super(adam, self).__init__(lr)
        self._b1 = beta1
        self._b2 = beta2
        self._eps = eps
        self._t = 0
        self._m = 0
        self._v = 0


    def optimize(self, model):
        model_grads = self._get_grads(model)
        model_grads_step = self._computer_grads(model_grads)
        self._appley_grads(model, model_grads_step)

    def _get_grads(self, model):
        l_grads = []
        for curr_layer in model.layers:
            if curr_layer.description['trainable']:
                    for name in curr_layer.param_names:
                        if not model.gpu_backend:
                            l_grads.append(curr_layer.grads[name])
                        else:
                            l_grads.append(curr_layer.grads_cuda[name])

        return np.array(l_grads) if not model.gpu_backend else cp.array(l_grads)

    def _computer_grads(self,  grads):
        self._t += 1
        self._m += (1 - self._b1) * (grads - self._m)
        self._v += (1 - self._b2) * (grads ** 2 - self._v)

        _m = self._m / (1 - self._b1 ** self._t)
        _v = self._v / (1 - self._b2 ** self._t)

        grad_step = self.lr * _m / (_v ** 0.5 + self._eps)
        return grad_step

    def _appley_grads(self, model, grad_step):
        idx = 0
        for curr_layer in model.layers:
            if curr_layer.description['trainable']:
                    for name in curr_layer.param_names:
                        if not model.gpu_backend:
                            curr_layer.params[name] -= grad_step[idx]
                            idx += 1
                        else:
                            curr_layer.params_cuda[name] -= grad_step[idx]
                            idx += 1


