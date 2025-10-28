import numpy as np


class ReLU:
    def forward(self, inputs):
        self.output = np.maximum(0, inputs)


def softmax(x):
    exp_vals = []
    norm_vals = []
    for output in x:
        exp_vals.append(np.exp(output))
    norm_div = sum(exp_vals)
    for val in exp_vals:
        norm_vals.append(val / norm_div)