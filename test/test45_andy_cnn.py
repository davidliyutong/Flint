from models import CNN5
from core.Optimizers import sgd, bgd
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
            if cp.argmax(res) == cp.argmax(test_label):
                cnt_correct += 1
            cnt_tot += 1
    else:
        for i in range(num_of_sample):
            test_input = test_inputs[i:i + 1]
            test_label = test_labels[i]
            res = model.forward_prop(test_input)
            if np.argmax(res) == np.argmax(test_label):
                cnt_correct += 1
            cnt_tot += 1

    acc = cnt_correct / cnt_tot
    print('[ accuracy ]: ', acc * 100)
    return acc


fashion_mnist = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

train_images = np.expand_dims(train_images / 255, axis=-1)
test_images = np.expand_dims(test_images / 255, axis=-1)
train_labels = one_hot_f(train_labels, num_classes=10)
test_labels = one_hot_f(test_labels, num_classes=10)

CNN5.compile()
try:
    CNN5.restore('./CNN5_cuda.param')
except:
    pass

train_iterator = batch_iterator(batch_sz=16)
optimizer = bgd(0.1)
for i in range(100):
    optimizer.fit(CNN5, train_images, train_labels, train_iterator, epoch=1)
    test(CNN5, test_images, test_labels)
    CNN5.save('CNN5_cuda')
