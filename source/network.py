from layer import Layer
from activations import ReLU
from data import create_data

import matplotlib.pyplot as plt
import numpy as np

X, y = create_data(100, 3)

layer1 = Layer(2, 5)
activation1 = ReLU()

layer1.forward(X)
activation1.forward(layer1.output)
