import numpy as np
from itertools import count
from activation_functions import *
from loss_functions import *


class Layer:
    counter = count(0)

    def __init__(self, m, n):
        """

        Each row of W represents a neuron

        :param m: the dimension of the input. x ∈ R^m
        :param n: the dimension of the output z ∈ R^n
        """

        self.W = np.random.normal(0.0, 1. / m, (m, n))
        self.b = np.random.normal(0.0, 1., (n, 1))
        self.A = None
        self.dLdb = None
        self.dLdW = None
        self.id = next(Layer.counter)

    def forward(self, A):
        """
        Takes in a batch of activations from the previous layer ans returns a batch of pre-activations Z

        Notes
        -----
        W is (m x n)
        W.T @ A => (n x b)
        W.T @ A => (n x b) + (n x 1) => broadcasting => n x b

        :param A: (m x b) : b column vectors each with dimension m
        :return:
        """
        self.A = A
        return self.W.T @ A + self.b

    def backward(self, dLdZ):
        """Given the change in loss due to this layer's preactivation, return the loss due to the activation

        Notes
        -----
        Our eventual goal is to compute the change in the loss due to W and b

        Parameters
        ----------
        dLdZ : ndarray
            (n x b) The change in loss due to this layer's preactivation, given by the backward method of
            the activation module from this layer

        Returns
        -------
        dLdA : ndarray
            (m x b) The change in loss due to the activation from the previous layer
            This is the input to be fed to the backward method of the activation method
            of the previous layer


        """
        _, b = dLdZ.shape
        self.dLdW = self.A @ dLdZ.T  # (m x b) @ (b x n) => (m x n)
        self.dLdb = dLdZ @ np.ones((b, 1))  # (n x b) @ (b x 1) => (n x 1)
        return self.W @ dLdZ  # (m x b)

    def sgd_step(self, learning_rate):
        """

        Notes
        -----
        
        Parameters
        ----------

        Returns
        -------

        """
        self.W -= (learning_rate * self.dLdW)
        self.b -= (learning_rate * self.dLdb)

    def __repr__(self):
        return repr(self.W)


class NeuralNetwork:
    activation_functions = {
        'tanh': Tanh,
        'sigmoid': Sigmoid,
        'softmax': Softmax,
        'relu': RELU,
        'param_relu': ParametrizedRELU,
        'leaky_relu': LeakyRELU,
        'linear': Linear

    }
    loss_functions = {
        'squared': SquaredError,
        'hinge': HingeLoss,
        'nll': NegativeLogLikelihood
    }

    @staticmethod
    def check_loss_id_string(loss_function):
        if loss_function not in NeuralNetwork.loss_functions:
            raise KeyError(f"{loss_function} nof found. "
                           f"Possible ids are {NeuralNetwork.loss_functions.keys()}")

    @staticmethod
    def check_activation_id_strings(activations):
        for activation_id in activations:
            if activation_id not in NeuralNetwork.activation_functions:
                raise KeyError(f"{activation_id} nof found. "
                               f"Possible ids are {NeuralNetwork.activation_functions.keys()}")

    @staticmethod
    def check_dimensions(layer_dims):
        if len(layer_dims) < 2:
            raise ValueError("The number of layer dimensions must be >= 2")

    @staticmethod
    def check_types(layer_dims, output_dim, activation_functions, loss_function):
        activation_types_allowed = (ActivationFunction, str)
        for activation_function in activation_functions:
            if not isinstance(activation_function, activation_types_allowed):
                raise ValueError(f" {activation_function} failed type check. "
                                 f"All the activation functions must {activation_types_allowed}")
        loss_function_types_allowed = (LossFunction, str)
        if not isinstance(loss_function, loss_function_types_allowed):
            raise ValueError(f" {loss_function} failed type check. "
                             f"All the loss function must {loss_function_types_allowed}")

        # to make things easier, we assert that output dim must be a numpy array of type int
        if not isinstance(layer_dims, np.ndarray):
            raise ValueError()

    def __init__(self, *, layer_dims, output_dim, activation_functions, loss_function):
        assert len(layer_dims) == len(activation_functions)
        NeuralNetwork.check_activation_id_strings(activation_functions)
        NeuralNetwork.check_loss_id_string(loss_function)
        NeuralNetwork.check_dimensions(layer_dims)
        NeuralNetwork.check_types(layer_dims, output_dim, activation_functions, loss_function)

        layers = []
        activations = []

        layer_dims += [output_dim]
        for m, n, activation_function in zip(layer_dims[:-1], layer_dims[1:], activation_functions):
            layers.append(Layer(m, n))
            activations.append(NeuralNetwork.activation_functions[activation_function])

        self.loss = NeuralNetwork.loss_functions[loss_function]
        self.layers = np.array(layers)
        self.activation_functions = np.array(activations)

    def fit(self, X_train, y_train, n_epochs):
        pass

    def sgd_neural_net(self, X_train, y_train, T, learning_rate):
        n, = y_train.shape
        for i in range(T):
            r = np.random.randint(0, n)
            A = X_train[r, :]
            y = y_train[r: r+1]
            for layer_idx, layer, activation in enumerate([self.layers, self.activation_functions]):
                Z = layer.forward(A)
                A = activation(Z)
            loss = self.loss.fn(A, y_train)
            # for layer_idx, layer, activation in reversed(enumerate([self.layers, self.activation_functions])):
            #
            #     layer.sgd_step(learning_rate)

    def predict(self):
        pass

    def __repr__(self):
        return repr(self.layers)


if __name__ == '__main__':
    nn = NeuralNetwork(layer_dims=[3, 5, 6],
                       output_dim=5,
                       activation_functions=['relu', 'relu', 'relu'],
                       loss_function='nll')
