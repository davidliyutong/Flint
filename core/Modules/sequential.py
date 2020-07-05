import numpy as np
import json
import time
import cupy as cp


def str_to_json(s):
    return json.loads(s)


def json_to_str(obj):
    return json.dumps(obj)


def np_ary_to_list(ary: np.array):
    l = ary.tolist()
    return l


def list_to_np_ary(l: list):
    ary = np.array(l, dtype=np.float32)
    return ary


def npdict_to_ldict(npdict: dict):
    ldict = {}
    for key in npdict.keys():
        ldict[key] = np_ary_to_list(npdict[key])

    return ldict


def ldict_to_npdict(ldict: dict):
    npdict = {}
    for key in ldict.keys():
        npdict[key] = list_to_np_ary(ldict[key])

    return npdict


class sequential(object):
    def __init__(self, *args, loss_f=None):
        self.layers = []
        self.description = {'type': 'Modules',
                            'layers': [],
                            'params': []}
        self.loss_f = loss_f
        self._compose_layers(args)
        self.gpu_backend = False

    def set_loss(self, loss_f):
        self.loss_f = loss_f

    def _num_layers(self):
        return len(self.description['layers'])

    def _compose_layers(self, input_layers):
        if len(input_layers) == 0:
            raise IndexError
        else:
            for curr_layer in input_layers:
                self.add(curr_layer)

    def add(self, new_layer):
        assert new_layer.description['type'] == 'layer'
        self.layers.append(new_layer)

    def get_description(self):
        string = json.dumps(self.description['layers'], indent=4, ensure_ascii=False)
        return string

    def compile(self):
        assert self.loss_f is not None
        assert self.layers[0].is_compiled

        prev_output_shape = self.layers[0].output_shape
        for curr_layer in self.layers:
            if prev_output_shape is None:
                raise RuntimeError
            if not curr_layer.is_compiled:
                curr_layer.compile(prev_output_shape)

            self.description['layers'].append(curr_layer.description)
            self.description['params'].append([])

            prev_output_shape = curr_layer.output_shape

    def _update_state_dicts(self):
        for i in range(self._num_layers()):
            self.description['params'][i] = npdict_to_ldict(self.layers[i].get_params())

    def _save_description(self, filename=None):
        ext = '.param'
        if filename is None:
            filename = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime()) + '_state_dict' + ext
        else:
            assert isinstance(filename, str)
            if filename.split('.')[-1] != ext.split('.')[-1]:
                filename = filename + ext

        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(self.description, f, indent=4, ensure_ascii=False)

    def save(self, filename=None):
        self._update_state_dicts()
        self._save_description(filename)
        print('[ msg ] Successfully saved parameters ! ')

    def _verify_loaded_description(self, new_description):
        try:
            if json.dumps(new_description['layers']) == json.dumps(self.description['layers']):
                return True
            else:
                return False
        except AttributeError:
            print('Wrong format !')

    def restore(self, filename):
        with open(filename, 'r', encoding='utf-8', errors='ignore') as f:
            new_description = json.load(f, strict=False)

        if not self._verify_loaded_description(new_description):
            print('[ msg ] Fail to restore parameters ! ')
            raise SyntaxError

        for i in range(self._num_layers()):
            new_state_dict = new_description['params'][i]
            assert isinstance(new_state_dict, dict)
            converted_state_dict = ldict_to_npdict(new_state_dict)
            self.layers[i].restore(converted_state_dict)

        print('[ msg ] Successfully restored parameters ! ')

    def cpu(self):
        if self.gpu_backend:
            for curr_layer in self.layers:
                curr_layer.cpu()
            self.gpu_backend = False
        else:
            pass

    def cuda(self):
        if not self.gpu_backend:
            for curr_layer in self.layers:
                curr_layer.cuda()
            self.gpu_backend = True
        else:
            pass

    def _backward_prop_cpu(self, x, labels):
        loss = self.loss_f.loss(x, labels)
        grad = self.loss_f.grad(x, labels)

        for curr_layer in reversed(self.layers):
            grad = curr_layer.backward(grad)

        return loss

    def _forward_prop_cpu(self, x):
        output = x
        for curr_layer in self.layers:
            output = curr_layer.forward(output)

        return output

    def _backward_prop_cuda(self, x, labels):
        loss_cuda = self.loss_f.loss_cuda(x, labels)
        grad_cuda = self.loss_f.grad_cuda(x, labels)

        for curr_layer in reversed(self.layers):
            grad_cuda = curr_layer.backward_cuda(grad_cuda)

        return float(cp.asnumpy(loss_cuda))

    def _forward_prop_cuda(self, x):
        output_cuda = x
        for curr_layer in self.layers:
            output_cuda = curr_layer.forward_cuda(output_cuda)

        return output_cuda

    def backward_prop(self, x, labels):
        if self.gpu_backend:
            return self._backward_prop_cuda(x, labels)
        else:
            return self._backward_prop_cpu(x, labels)

    def forward_prop(self, x):
        if self.gpu_backend:
            return self._forward_prop_cuda(x)
        else:
            return self._forward_prop_cpu(x)

    def val(self, x):
        return self.forward_prop(x)
