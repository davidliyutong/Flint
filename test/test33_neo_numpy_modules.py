import numpy as np
from core.Layers import conv2d, max_pool, flatten, relu, dense
from core.Functions.loss import softmax_cross_entropy

cv2d = conv2d(input_shape=(28, 28, 3), output_ch=3)
input = np.random.randn(3, 28, 28, 3)
lbl = [0, 1, 0, 0, 0, 0, 0, 0, 0, 0]
a = cv2d.forward(input)
print(a.shape)
cv2d.backward(a)
print(cv2d.grads)

maxp = max_pool(kernel_sz=3)
maxp.compile(cv2d.output_shape)
print(maxp.description)
b = maxp.forward(a)
a_grad = maxp.backward(b)
print(a_grad)

rl = relu()
rl.compile(maxp.output_shape)
print(rl.description)
c = rl.forward(b)
b_grad = rl.backward(c)
print(b_grad)

flt = flatten()
flt.compile(rl.output_shape)
print(flt.description)
d = flt.forward(c)
c_grad = flt.backward(d)
print(c_grad)

ds = dense(output_shape=10)
ds.compile(flt.output_shape)
print(ds.description)
e = ds.forward(d)
d_grad = ds.backward(e)
print(d_grad)

ls = softmax_cross_entropy()
f = ls.loss(e, lbl)
print(f)
e_grad = ls.grad(e, lbl)
print(e_grad)
