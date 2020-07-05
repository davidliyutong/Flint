from models import CNN4
from core.Optimizers import sgd, adam
from core.Functions import one_hot_f
import numpy as np
from tensorflow import keras
from core.Dataloader import batch_iterator
import cupy as cp


def test(model, test_inputs, test_labels):
    num_of_sample = test_inputs.shape[0]
    cnt_correct, cnt_tot = 0, 0
    if model.gpu_backend:
        test_inputs = cp.array(test_inputs)
        test_labels = cp.array(test_labels)
    for i in range(num_of_sample):
        test_input = test_inputs[i:i + 1]
        test_label = test_labels[i]
        res = model.forward_prop(test_input)
        if np.argmax(res) == np.argmax(test_label):
            cnt_correct += 1
        cnt_tot += 1

    acc = cnt_correct / cnt_tot
    print('[ accuracy ]: ', acc)
    return acc


fashion_mnist = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

train_images = np.expand_dims(train_images / 255, axis=-1)[0:6000]
test_images = np.expand_dims(test_images / 255, axis=-1)
train_labels = one_hot_f(train_labels, num_classes=10)[0:6000]
test_labels = one_hot_f(test_labels, num_classes=10)

CNN4.compile()
train_iterator = batch_iterator(batch_sz=16)
optimizer = adam(lr=1e-3)
for i in range(100):
    optimizer.fit(CNN4, train_images, train_labels, train_iterator, epoch=1)
    test(CNN4, train_images, train_labels)
