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


class Loss:
    def calculate(self, output, y):
        sample_losses = self.forward(output, y)
        batch_loss = np.mean(sample_losses)
        return batch_loss


class CCE(Loss):
    def forward(self, y_pred, y_true):
        samples = len(y_pred)
        y_pred_clip = np.clip(y_pred, 1e-7, 1 - 1e-7)

        if len(y_true.shape) == 1:
            conf_corrects = y_pred_clip[range(samples), y_true]
        elif len(y_true.shape) == 2:
            conf_corrects = np.sum(y_pred_clip * y_true, axis=1)

        neg_log_probs = -np.log(conf_corrects)
        return neg_log_probs
