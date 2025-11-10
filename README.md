<h1>A Small-Scale Neural Network, From Scratch in Numpy</h1>

Hello!

This is my first project on GitHub, and while it is a type of program I am familiar with (a neural network), I thought it would be a good exercise for teaching myself version control, as git is a system that I was not familiar with. I've also almost exclusively built NNs using libraries such as TF and PyTorch so I thought it would be fun to try and build one from the ground up, using only numpy for the math.

I plan on improving this neural network as time goes on, as well as writing similar versions in other languages that I know, in order to get a good sense of efficiency and to work on my ability to write fast code. It is a binary classifier, and can be trained to solve the XOR problem as seen in the example provided.

<h2>Architecture</h2>

This is a simple neural network, with 2 layers. The first has an input size of 2 (corresponding to 2 binary inputs), and a layer size of 4 neurons. The second has an input size of 4 (corresponding to the number of neurons in the previous layer), and a layer size of 1 neuron (corresponding to the single desired binary output). 

Each layer's output is then fed into a sigmoid activation function, $$\sigma(z) = \frac{1} {1 + e^{-z}}$$ .
This is a common activation function for binary classification problems, as it introduces non-linerality into the model as well as asymptotically approaching 0 as x increases to negaitve infinity and approaching 1 as x increases to infinity. This "smooths" outputs, making the numbers much easier to work with. It is also important during backpropagation, as it is differentiable at every point and is, as far as I am aware, the only function which is included in its derivative, which makes it computationally simple to model. 

Sigmoid is not perfect, however, and I encountered one of its main issues during the creation of this network. During backpropagation, the sigmoid function can experience something called the Vanishing Gradient Problem. While updating weights and biases via gradient descent, if the gradient is too small the updates become insignificant. This can lead to learning slowing down or (as I saw in my case) the network sometimes stopping learning all together. My architecture has this problem, as with a learning rate of 0.5 it took 50k epochs to train to 100% accuracy, an excessively high number. Increasing the learning rate to 1 remedied this problem, but the network is stil rather slow.

Loss for this network is calculated via Mean Squared Error (MSE). There are other common approaches to calculating loss for a network like this, the other most common being Binary Cross-Entropy (BCE). I decided to go with MSE over BCE for this architecture as it is the one with which I was more familiar, but I may swap them out at a later date. 
