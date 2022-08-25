from abc import ABC, abstractmethod


class OptimizerBase(ABC):
    def __init__(self, lr):
        pass


class SGD(OptimizerBase):
    pass


class AdaGrad(OptimizerBase):
    pass


class RMSProp(OptimizerBase):
    pass


class Adam(OptimizerBase):
    pass

