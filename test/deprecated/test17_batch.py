from models import Linear2
from sklearn import datasets
from core.Optimizers import sgd
from core.Functions import one_hot_f
import numpy as np

iris = datasets.load_iris()
iris_label_one_hot = one_hot_f(iris.target, num_classes=3)

Linear2.compile()
print(Linear2.val(np.transpose(iris.data)))
