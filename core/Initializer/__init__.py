import numpy as np


def compute_fans(shape):
    if len(shape) == 2:
        fan_in, fan_out = shape[0], shape[1]
    else:
        fan_in, fan_out = np.prod(shape[1:]), shape[0]
    return fan_in, fan_out


class initializer(object):
    def __call__(self, shape):
        return self.init(shape).astype(np.float32)

    def init(self, shape):
        raise NotImplementedError


class constant(initializer):
    def __init__(self, val):
        self._val = val

    def init(self, shape):
        return np.full(shape=shape, fill_value=self._val).astype(np.float32)


class zeros(constant):
    def __init__(self):
        super(zeros, self).__init__(0.0)


class xavieruniform(initializer):
    def __init__(self, gain=1.0):
        self._gain = gain

    def init(self, shape):
        fan_in, fan_out = compute_fans(shape)
        a = self._gain * np.sqrt(6.0 / (fan_in + fan_out))
        return np.random.uniform(low=-a, high=a, size=shape).astype(np.float32)
