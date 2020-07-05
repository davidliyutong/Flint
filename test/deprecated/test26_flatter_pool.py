from core.Layers import max_pool
from core.Layers import flatten
import numpy as np
import matplotlib.pyplot as plt

fl = flatten(input_dim=(28, 28, 16), batch_size=32)
print(fl.description)
A = np.random.randn(28, 28, 16, 32)
B = fl.forward(A)

mp = max_pool(input_dim=(28, 28, 16), kernel_sz=7, batch_size=32)
C = mp.forward(A)

D = A[:, :, 0, 0]
E = C[:, :, 0, 0]

plt.imshow(D)
plt.show()
plt.imshow(E)
plt.show()

F = mp.backward(C)
print(F)
