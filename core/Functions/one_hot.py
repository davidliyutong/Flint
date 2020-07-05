import numpy as np


def one_hot_f(ary, num_classes=1):
    assert type(ary) is np.ndarray
    ary = ary.astype(np.int32)

    if (ary.max() - ary.min()) >= num_classes:
        raise ValueError
    else:
        output = np.eye(num_classes)[ary]
        return output
