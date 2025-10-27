import matplotlib.pyplot as mplt
import numpy as np

np.random.seed(0)


# https://cs231n.github.io/neural-networks-case-study/
def create_data(num_points, num_classes):
    X = np.zeros((num_points * num_classes, 2))
    y = np.zeros(num_points * num_classes, dtype="uint8")
    for class_item in range(num_classes):
        ix = range(num_points * class_item, num_points * (class_item + 1))
        r = np.linspace(0.0, 1, num_points)
        t = (
            np.linspace(class_item * 4, (class_item + 1) * 4, num_points)
            + np.random.randn(num_points) * 0.2
        )
        X[ix] = np.c_[r * np.sin(t * 2.5), r * np.cos(t * 2.5)]
        y[ix] = class_item
    return X, y
