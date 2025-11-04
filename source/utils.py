import numpy as np

class Layer:
    def __init__(self, n_inputs, n_neurons):
        self.weights = 0.10 * np.random.randn(n_inputs, n_neurons)
        self.bias = np.zeros((1, n_neurons))

    def forward(self, inputs):
        self.output = np.dot(inputs, np.array(self.weights)) + self.bias


class ReLU:
    def forward(self, inputs):
        self.output = np.maximum(0, inputs)


class Softmax:
    def forward(self, inputs):
        exp_vals = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        probs = exp_vals / np.sum(exp_vals, axis=1, keepdims=True)
        self.output = probs
