import numpy as np


def sigmoid(inp):
    return 1 / (1 + np.exp(-inp))


def de_sigmoid(inp):
    x = sigmoid(inp)
    return x * (1 - x)


def tanh(inp):
    return 2 * sigmoid(2 * inp) - 1


def de_tanh(inp):
    return 1 - tanh(inp) ** 2


def softmax(inp):
    _inp = inp - np.max(inp, axis=-1, keepdims=True)
    _inp = np.exp(_inp)
    return _inp / np.sum(_inp, axis=-1, keepdims=True)


def one_hot_encode(labels, n_types):
    return np.eye(n_types)[labels]


def one_hot_decode(pred):
    return np.argmax(pred, axis=1)


def MSE_loss(pred, target):
    # 1/n * sum[(target_i-pred_i)**2,i]
    return np.mean((target - pred) ** 2), 2 * (target - pred)


def cross_entropy_loss(pred, target):
    """
    automatically softmax the input
    pred -= np.max(pred) to avoid overflow
    :param pred: a list that contain batch of predict result (no softmax)
    :param target: a list contain index of correct type
    :return:
    """

    # process pred to a ndarry which has 2 dimensions
    if not isinstance(pred, np.ndarray):
        pred = np.array(pred)
    if len(pred.shape) == 1:
        pred = np.expand_dims(pred, axis=0)

    # one-hot encode target
    target = one_hot_encode(target, pred.shape[1])
    # target * Log[Softmax[pred]]

    tmp = softmax(pred)
    tmp[np.logical_or(np.isnan(tmp), np.isinf(tmp))] = 0
    return -np.sum(target * np.log(tmp), axis=-1, keepdims=True), tmp - target
