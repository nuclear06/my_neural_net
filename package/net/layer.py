from My_Mnist_Net.package.net.base import Layer, Parameter
from My_Mnist_Net.package.net.math import *


class Liner(Layer):
    def __init__(self, inp_dim, outp_dim, init_func=None):
        # X @ W + b
        self.inp_dim = inp_dim
        self.outp_dim = outp_dim

        if init_func:
            # 1 for batch_size
            self.weight = init_func(inp_dim, outp_dim).reshape(1, inp_dim, outp_dim)
            self.bias = init_func(outp_dim).reshape(1, 1, outp_dim)
        else:
            self.weight = np.random.randn(1, inp_dim, outp_dim) / np.sqrt(inp_dim / 2)
            self.bias = np.random.randn(1, 1, outp_dim) / np.sqrt(inp_dim / 2)
        self.weight, self.bias = Parameter(self.weight), Parameter(self.bias)

        # store previous input data for backward pass
        self.pre_inp = None

    def forward(self, inp):
        """
        :param inp:
        :return:
        """
        # (batchSize,1,realData)
        _inp = inp.reshape(-1, 1, self.weight.value.shape[1])
        self.pre_inp = _inp
        return _inp @ self.weight.value + self.bias.value

    def backward(self, grad):

        # dv of input is grad @ WT
        # (1, ?, outp) @ (1, outp, inp) = (1, ?. inp)
        dv_pre = grad * np.sum(self.weight.value, axis=0, keepdims=True).transpose(0, 2, 1)

        # dv of weight is XT @ grad
        # swap (batch, 1, inp) to (batch, inp, 1) repeat to (batch, inp, outp)
        # grad is (1, 1, outp) repeat to (1, inp, outp)
        self.weight.dv = (self.pre_inp.transpose(0, 2, 1).repeat(self.outp_dim, axis=-1) *
                          grad.repeat(self.inp_dim, axis=1))
        # dv of bias is grad
        self.bias.dv = grad

        # Compute mean
        self.weight.dv = np.mean(self.weight.dv, axis=0, keepdims=True)
        self.bias.dv = np.mean(self.bias.dv, axis=0, keepdims=True)

        return dv_pre


class SigmoidLayer(Layer):
    # f(x)=1/(1+e^x)
    # f'(x)=f(x)(1-f(x))
    def __init__(self):
        self.pre_inp = None

    def forward(self, inp):
        self.pre_inp = inp
        return sigmoid(inp)

    def backward(self, grad):
        return grad * de_sigmoid(self.pre_inp)


class TanhLayer(Layer):
    # f(x)=tanh(x)
    def __init__(self):
        self.pre_inp = None

    def forward(self, inp):
        self.pre_inp = inp
        return tanh(inp)

    def backward(self, grad):
        return grad * de_tanh(self.pre_inp)


class LeakyReLULayer(Layer):
    # f(x) = leak*x , if x<0
    #        x      , if x>=0
    def __init__(self, leak=0.01):
        self.prev_inp = None
        self.leak = leak

    def forward(self, inp):
        self.prev_inp = inp
        return np.maximum(inp, self.leak * inp)

    def backward(self, grad):
        self.prev_inp[self.prev_inp < 0] = self.leak
        self.prev_inp[self.prev_inp > 0] = 1
        return grad * self.prev_inp


class ReLULayer(LeakyReLULayer):
    def __init__(self):
        # f(x) = 0  , if x<0
        #        x  , if x>=0
        super().__init__(0)
