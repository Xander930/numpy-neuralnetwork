from data import create_data
from utils import CCE, Layer, ReLU, Softmax

X, y = create_data(100, 3)

layer1 = Layer(2, 3)
layer2 = Layer(3, 3)

activ1 = ReLU()
activ2 = Softmax()

layer1.forward(X)
activ1.forward(layer1.output)

layer2.forward(activ1.output)
activ2.forward(layer1.output)

loss_f = CCE()
loss = loss_f.calculate(activ2.output, y)

print("Loss:", loss)