import numpy as np

from My_Mnist_Net.package.net.base import *


class SGD(Optimizer):
    def __init__(self, lr, gamma=0.9, decay=1e-8):
        self.params = []
        self.lr = lr
        self.gamma = gamma
        self.v = []  # Momentum
        self.epoch = 0  # use for decay of lr
        self.decay = decay

    def link_parameters(self, params):
        self.params = params
        self.v = [np.zeros_like(p) for p in params]

    def step(self):
        lr = self.lr / (1 + self.epoch * self.decay)
        self.epoch += 1
        for idx, p in enumerate(self.params):
            self.v[idx] = self.gamma * self.v[idx] + lr * p.dv
            p.value -= self.v[idx]
