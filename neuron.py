import numpy as np

inputs = []
weights = []
bias = []

outputs_layer1 = np.dot(inputs, np.array(weights).T) + bias

outputs_layer2 = np.dot(outputs_layer1, np.array(weights).T) + bias