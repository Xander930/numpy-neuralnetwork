import numpy as np


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))


def sigmoid_prime(x):
    s = sigmoid(x)
    return s * (1 - s)
