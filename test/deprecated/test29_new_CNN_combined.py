from models import CNN1, CNN2, CNN3
from sklearn import datasets
from core.Optimizers import sgd, bgd
from core.Functions import one_hot_f
import numpy as np
from tensorflow import keras


def test(model, test_inputs, test_labels):
    num_of_sample = test_inputs.shape[0]
    cnt_correct, cnt_tot = 0, 0
    for i in range(num_of_sample):
        test_input = np.expand_dims(np.transpose(test_inputs[i]), axis=-1)
        test_label = np.expand_dims(np.transpose(test_labels[i]), axis=-1)
        res = model.forward_prop(test_input)
        if np.argmax(res) == np.argmax(test_label):
            cnt_correct += 1
        cnt_tot += 1

    return cnt_correct / cnt_tot


fashion_mnist = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

# origin = np.random.randn(3, 28, 28).astype(np.float32)
# input_image = train_images[0, :, :].reshape(1,28,28)
train_images = np.expand_dims(train_images / 255, axis=1)
test_images = np.expand_dims(test_images / 255, axis=1)
train_labels = one_hot_f(train_labels, num_classes=10)
test_labels = one_hot_f(test_labels, num_classes=10)

small_train_images = train_images[0:10000]
small_train_labels = train_labels[0:10000]
small_test_images = train_images[0:100]
small_test_labels = train_labels[0:100]

CNN2.compile()
acc = test(CNN2, small_test_images, small_test_labels)
print(acc)
optimizer1 = sgd(0.001)

for i in range(5):
    optimizer1.fit(CNN2, train_images, train_labels, epoch=1, batch_size=1)
    CNN2.set_batch_size(1)
    acc = test(CNN2, small_test_images, small_test_labels)
    print(acc)
