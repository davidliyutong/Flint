from models import Linear2
from sklearn import datasets
from core.Optimizers import sgd
from core.Functions import one_hot_f
import numpy as np


def test(model, datas, labels, tagets):
    acc = 0
    for i in range(len(datas)):
        sample = datas[i]
        if len(sample.shape) < 2:
            sample = np.expand_dims(sample, axis=1)
        res = model.val(sample)
        if tagets[i] == np.argmax(res, axis=0):
            acc += 1

    print('[ acc ]: ', acc / len(datas))


iris = datasets.load_iris()
iris_label_one_hot = one_hot_f(iris.target, num_classes=3)

print(Linear2.description)
Linear2.compile()
optimizer = sgd(0.009)
# test(Linear2, iris.data, iris_label_one_hot, iris.target)
optimizer.fit(Linear2, iris.data, iris_label_one_hot, epoch=2000)
test(Linear2, iris.data, iris_label_one_hot, iris.target)
