class loss(object):
    def __init__(self):
        self.last_input = None
        self.grads = {}
        self.grads_cuda = {}

    def loss(self, x, labels):
        raise NotImplementedError

    def grad(self, x, labels):
        raise NotImplementedError

    def loss_cuda(self, x, labels):
        raise NotImplementedError

    def grad_cuda(self, x, labels):
        raise NotImplementedError
