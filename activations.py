inputs = []

def ReLU(inputs):
    output = []
    for input in inputs:
        output.append(max(0, input))