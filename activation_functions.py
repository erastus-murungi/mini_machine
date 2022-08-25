from abc import ABC, abstractmethod
import numpy as np


class ActivationFunction(ABC):
    def __init__(self):
        self.A = None

    @abstractmethod
    def fn(self, Z):
        pass

    @abstractmethod
    def gradient(self, V):
        pass

    def forward(self, Z):
        """
        :param Z: preactivation of layer l
        :return: the activation of layer l
        """
        self.A = self.fn(Z)
        return self.A

    def backward(self, dLdA):  # uses stored self.A
        """

        Notes
        -----
        dLdZ = dLdA * dAdZ by the chain rule
        dAdZ is just the derivative of tanh due to the pre-activation Z


        :param dLdA: dldA from layer l + 1
        :return: dLdZ to be used in layer l
        """

        return dLdA * self.gradient(self.A)


class RELU(ActivationFunction):
    def __init__(self):
        super().__init__()

    def fn(self, Z):
        return np.maximum(0, Z)

    def gradient(self, V):
        return (V != 0).astype(np.float64)


class Softmax(ActivationFunction):
    def fn(self, Z):
        return np.exp(Z) / np.sum(np.exp(Z), axis=0)

    def gradient(self, V):
        return np.ones_like(V)

    @staticmethod
    def most_likely(y):
        return np.argmax(y, axis=0)


class ParametrizedRELU(ActivationFunction):
    def __init__(self, alpha=0.01):
        super(ParametrizedRELU, self).__init__()
        self.alpha = alpha

    def fn(self, Z):
        return np.where(Z > 0, Z, Z * self.alpha)

    def gradient(self, V):
        return np.where(V < 0.0, self.alpha, 1.0)


class LeakyRELU(ParametrizedRELU):
    def __init__(self):
        super().__init__(alpha=0.01)


class Linear(ActivationFunction):
    def fn(self, Z):
        return Z

    def gradient(self, V):
        return np.ones_like(V)


class Tanh(ActivationFunction):
    def fn(self, Z):
        return np.tanh(Z)

    def gradient(self, V):
        """
        d tanh(z) / dz = 1 - tanh(z)**2
        """
        return 1. - (V * V)


class Sigmoid(ActivationFunction):

    def fn(self, Z):
        return 1. / (1. + np.exp(-Z))

    def gradient(self, V):
        fn_V = self.fn(V)
        return fn_V * (1. - fn_V)


__all__ = [RELU, Sigmoid, Tanh, LeakyRELU, ParametrizedRELU, Softmax, Linear]

