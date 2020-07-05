from .optimizer import optimizer
import time


class sgd(optimizer):
    def __init__(self, lr=1e-4):
        super(sgd, self).__init__(lr)

    def set_lr(self, lr):
        # set learning rate
        self.lr = lr

    def optimize(self, model):
        for curr_layer in model.layers:
            if curr_layer.description['trainable']:
                for name in curr_layer.param_names:
                    if not model.gpu_backend:
                        curr_layer.params[name] -= curr_layer.grads[name] * self.lr
                    else:
                        curr_layer.params_cuda[name] -= curr_layer.grads_cuda[name] * self.lr