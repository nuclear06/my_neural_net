import logging
from abc import abstractmethod


class Layer:
    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def forward(self, inp):
        pass

    @abstractmethod
    def backward(self, grad):
        pass


class Parameter:
    def __init__(self, value):
        self.value = value
        self.dv = None


class Optimizer:
    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def link_parameters(self, layers):
        pass

    @abstractmethod
    def step(self):
        pass


class Net:
    def __init__(self, loss_func, optimizer: Optimizer, layers=[]):
        self.loss_func = loss_func
        self.optimizer = optimizer
        self.layers = layers
        optimizer.link_parameters(self.parameters)

    def train(self, data, target):
        layer: Layer

        loss, grad = self.loss_func(self.predict(data), target)
        for layer in reversed(self.layers):
            grad = layer.backward(grad)
        self.optimizer.step()
        return loss

    def predict(self, data):
        layer: Layer

        for layer in self.layers:
            data = layer.forward(data)
        return data

    @property
    def parameters(self):
        for layer in self.layers:
            return [p for p in vars(layer).values() if isinstance(p, Parameter)]
