import numpy as np
import cupy as cp


class batch_iterator:
    def __init__(self, batch_sz=32, shuffle=True):
        self.batch_sz = batch_sz
        self.shuffle = shuffle
        self.gpu_backend = False

    def __call__(self, inputs, labels):
        self.num_inputs = len(inputs)
        slices = np.arange(0, self.num_inputs, self.batch_sz)
        if self.shuffle:
            new_idx = np.arange(self.num_inputs)
            np.random.shuffle(new_idx)
            inputs = inputs[new_idx]
            labels = labels[new_idx]

        for slice in slices:
            end_slice = slice + self.batch_sz
            batch_ipt = inputs[slice: end_slice]
            batch_lbl = labels[slice: end_slice]
            if self.gpu_backend:
                batch_ipt_gpu = cp.asarray(batch_ipt.astype(np.float32))
                batch_lbl_gpu = cp.asarray(batch_lbl.astype(np.float32))
                yield {"inputs": batch_ipt_gpu, "labels": batch_lbl_gpu}
            else:
                yield {"inputs": batch_ipt, "labels": batch_lbl}

    def cuda(self):
        if not self.gpu_backend:
            self.gpu_backend = True

    def cpu(self):
        if self.gpu_backend:
            self.gpu_backend = False


class batch_iterator_normalize(batch_iterator):
    def __inti__(self, batch_sz=32, shuffle=True):
        super(batch_iterator_normalize, self).__inti__(batch_sz, shuffle)

    def __call__(self, inputs, labels):
        inputs = inputs / np.max(inputs)
        super(batch_iterator_normalize, self).__call__(inputs, labels)
