import cupy as cp


class layer(object):
    def __init__(self):
        self.is_compiled = False
        self.description = {'type': 'layer'}
        self.gpu_backend = False

        self.params = {}
        self.param_names = {}
        self.params_cuda = {}
        self.grads = {}
        self.grads_cuda = {}
        self.shapes = {}

        self.output_shape = None
        self.input_shape = None
        self.input_ch = None
        self.output_sz = None

        self.last_input = None
        self.last_input_cuda = None
        self.last_input_col = None
        self.last_input_col_cuda = None
        self.output = None
        self.output_cuda = None

    def compile(self, *args):
        pass

    def restore(self, *args):
        pass

    def forward(self, *args):
        pass

    def forward_cuda(self, *args):
        pass

    def backward(self, *args):
        pass

    def backward_cuda(self, *args):
        pass

    def get_params(self):
        if self.gpu_backend:
            for name in self.param_names:
                self.params[name] = cp.asnumpy(self.params_cuda[name])
        return self.params

    def cuda(self):
        for name in self.param_names:
            self.params_cuda[name] = cp.array(self.params[name])
        self.gpu_backend = True

    def cpu(self):
        for name in self.param_names:
            self.params[name] = cp.asnumpy(self.params_cuda[name])
        self.gpu_backend = False
