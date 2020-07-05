from numba import cuda, float32
import numpy as np

while True:
    B = cuda.device_array(shape=(2000, 2000), dtype=np.float32)
