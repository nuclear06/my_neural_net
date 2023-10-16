import os
import gzip
import numpy as np


def load_dataset(folder_path, batch_size=100):
    files = [
        "train-labels-idx1-ubyte.gz",
        "train-images-idx3-ubyte.gz",
        "t10k-labels-idx1-ubyte.gz",
        "t10k-images-idx3-ubyte.gz",
    ]
    fps = [os.path.join(folder_path, file) for file in files]

    with gzip.open(fps[0], "rb") as lbpath:
        y_train = np.frombuffer(lbpath.read(), np.uint8, offset=8)

    with gzip.open(fps[1], "rb") as imgpath:
        x_train = np.frombuffer(imgpath.read(), np.uint8, offset=16).reshape(-1, 28, 28)

    with gzip.open(fps[2], "rb") as lbpath:
        y_test = np.frombuffer(lbpath.read(), np.uint8, offset=8)

    with gzip.open(fps[3], "rb") as imgpath:
        x_test = np.frombuffer(imgpath.read(), np.uint8, offset=16).reshape(-1, 28, 28)

    return (
        data_iter(batch_size, x_train, y_train),
        data_iter(batch_size, x_test, y_test),
    )


def data_iter(batch_size, data1, data2):
    data_len = len(data1)
    indices = list(range(data_len))
    np.random.shuffle(indices)
    for i in range(0, data_len, batch_size):
        _i = min(i + batch_size, data_len)
        yield data1[i:_i], data2[i:_i]
