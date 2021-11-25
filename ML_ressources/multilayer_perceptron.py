import numpy as np
import math

import os, sys, inspect

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

from utils.activation_function import choose_activation
from utils.loss_function import choose_loss

class MultilayerPerceptron():
    """ A fully connected neural network
    Params:
        sizes (List):
            list where each entry is the number of neurons in the respective layer
        n_iterations (Int):
            number of training iterations in the training process
        learning_rate (Float):
            step length for updatin the weights with gradient descent
        hid_func (String):
            name of the activation function in the hidden layer
        out_func (String):
            name of the activation function in the output layer
        loss_func (String):
            name of the loss function
        init_limit (Int):
            interval limit for the weight initialization
        stop_error (Float):
            if error is below this value the training will be stopped
    """

    def __init__(self, sizes,
                 n_iterations=3000,
                 learning_rate=0.1,
                 hid_func="tanh",
                 out_func="identity",
                 loss_func="square loss",
                 init_limit=1,
                 stop_error=10.0e-5):

        self.num_layers = len(sizes)
        self.sizes = sizes
        self.n_iterations = n_iterations
        self.learning_rate = learning_rate
        self.hidden_activation = choose_activation(hid_func)
        self.out_activation = choose_activation(out_func)
        self.loss = choose_loss(loss_func)
        self.init_limit = init_limit
        self.stop_error = stop_error

    def _initialize_weights(self, X, y):

        self.biases = [np.random.uniform(-self.init_limit, self.init_limit, (1, y)) for y in self.sizes[1:]]
        self.weights = [np.random.uniform(-self.init_limit, self.init_limit, (x, y))
                        for x, y in zip(self.sizes[:-1], self.sizes[1:])]

    def _forward_propagation(self, X, Y):
        y_pred = np.ones_like(Y)

        # loop through each input
        for i, x in enumerate(X):
            x = np.expand_dims(x, axis=1)
            count = 1
            # loop through each layer
            for b, w in zip(self.biases, self.weights):
                count += 1
                inp = np.dot(w.T, x) + b.T
                if count < self.num_layers:
                    x = self.hidden_activation(inp)
                else:
                    # output layer
                    y_pred[i] = self.out_activation(inp)

        return y_pred

    def _compute_gradients(self, x, y):
        """Return a tuple ``(nabla_b, nabla_w)`` representing the
        gradient for the cost function C_x.  ``nabla_b`` and
        ``nabla_w`` are layer-by-layer lists of numpy arrays, similar
        to ``self.biases`` and ``self.weights``."""

        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]

        # feedforward
        act = x
        activations = [x] # list to store all the activations, layer by layer
        inputs = [] # list to store all the z vectors, layer by layer

        count = 1
        # loop through each layer
        for b, w in zip(self.biases, self.weights):
            count += 1
            inp = np.dot(w.T, act) + b.T
            inputs.append(inp)

            if count < self.num_layers:
                act = self.hidden_activation(inp)
                activations.append(act)
            else:
                # output layer
                act = self.out_activation(inp)
                activations.append(act)

        # backward pass
        # compute error gradient
        error = np.atleast_2d(self.loss.gradient(y, act[-1]))
        # compute delta of output layer
        z = inputs[-1]
        delta = self.out_activation.gradient(z)
        # compute nablas
        nabla_b[-1] = np.dot(error, delta)
        nabla_w[-1] = np.dot(np.dot(error, delta), activations[-2].transpose()).transpose()

        # loop through hidden layers
        for l in range(2, self.num_layers):
            z = inputs[-l]
            act_grad = self.hidden_activation.gradient(z)
            # compute delta at consequent neuron
            delta = np.multiply(np.dot(self.weights[-l+1], delta), act_grad).transpose()
            # sum changes for bias (only one node)
            nabla_b[-l] = np.dot(error, delta)
            nabla_w[-l] = np.dot(error, np.dot(activations[-l-1], delta))
        return (nabla_b, nabla_w)



    def fit(self, X, Y):

        self._initialize_weights(X, Y)

        error = []

        for i in range(self.n_iterations):

            # forward pass
            y_pred = self._forward_propagation(X, Y)

            # calculate and save mean squared error
            error.append(self.loss(Y, y_pred).mean())

            # compute gradients
            nabla_b = [np.zeros(b.shape) for b in self.biases]
            nabla_w = [np.zeros(w.shape) for w in self.weights]
            for x, y in zip(X, Y):
                # compute and add for each training sample
                delta_nabla_b, delta_nabla_w = self._compute_gradients(np.atleast_2d(x), y)
                nabla_b = [nb + dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
                nabla_w = [nw + dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]

            # update weights and biases
            self.weights = [w - self.learning_rate * nw
                            for w, nw in zip(self.weights, nabla_w)]
            self.biases = [b - self.learning_rate * nb
                           for b, nb in zip(self.biases, nabla_b)]

            if error[-1] < self.stop_error:
                break

        return error

def main():
    X = np.array([[0.5503],
     [0.9206],
     [0.5359],
     [0.6081],
     [0.0202],
     [0.8545],
     [0.2357],
     [0.4847],
     [0.3996],
     [0.1957]])

    y = np.array([[-0.5894],
     [-0.2507],
     [-0.0468],
     [-0.3402],
     [ 0.2857],
     [-1.0683],
     [ 0.8605],
     [-0.0801],
     [ 0.6837],
     [ 1.185 ]])

    sizes = [1, 3, 1]

    mlp = MultilayerPerceptron(sizes, init_limit=0.05, learning_rate=0.05)
    error = mlp.fit(X, y)
    print(f"Final Error: {error[-1]}")



if __name__ == "__main__":
    main()