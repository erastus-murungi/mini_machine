from abc import ABC, abstractmethod
import numpy as np


class LossFunction(ABC):
    @abstractmethod
    def fn(self, predicted, actual):
        pass

    @abstractmethod
    def gradient(self, predicted, actual, pre_activation, activation_function):
        pass


class NegativeLogLikelihood(LossFunction):
    def fn(self, predicted, actual):
        return -((actual * np.log(predicted)) + ((1 - actual) * np.log(1 - predicted)))

    def gradient(self, predicted, actual, pre_activation, activation_function):
        pass


class NeuralNegativeLogLikelihood(LossFunction):
    def __init__(self):
        self.y_pred = self.y = None

    def forward(self, predicted, actual):
        self.y_pred = predicted
        self.y = actual
        return self.fn(predicted, actual)

    def backward(self):
        return self.gradient(self.y_pred, self.y, None, None)

    def fn(self, predicted, actual):
        eps = np.finfo(float).eps
        return np.float64(np.sum(-actual * np.log(predicted + eps)))

    def gradient(self, predicted, actual, pre_activation, activation_function):
        return predicted - actual


class SquaredError(LossFunction):
    def __init__(self):
        super(SquaredError, self).__init__()

    def fn(self, predicted, actual):
        return (predicted - actual) ** 2

    def gradient(self, actual, predicted, pre_activation, activation_function):
        return 2 * (actual - predicted) * activation_function.gradient(pre_activation)


class HingeLoss(LossFunction):
    def fn(self, predicted, actual):
        pass

    def gradient(self, actual, predicted, pre_activation, activation_function):
        raise ValueError(f"{self.__class__.__name__} has no gradient!")


