import numpy as np
from datasets import load_dataset
from tinygrad import Tensor, nn
from tinygrad.engine.jit import TinyJit

from learner import Learner, MetricsCB, TqdmCB, TrainCB
from loader import DataLoaders


class TinyMLP:
    def __init__(self, in_features=28 * 28, hidden=128, out_features=10):
        self.l1 = nn.Linear(in_features, hidden)
        self.l2 = nn.Linear(hidden, out_features)

    def __call__(self, x):
        return self.l2(self.l1(x).relu())


def transforms(batch: dict[str, np.ndarray]):
    x, y = "image", "label"
    return Tensor(batch[x]).reshape(-1, 28 * 28), Tensor(batch[y])


def loss_func(preds, y):
    return preds.cross_entropy(y)


BATCH_SIZE = 128
LR = 1e-3


def main():
    ds = load_dataset("zalando-datasets/fashion_mnist")
    ds = ds.with_format("numpy")

    dls = DataLoaders.from_dd(ds, BATCH_SIZE, transform=transforms)
    model = TinyMLP()

    @TinyJit
    def accuracy(preds, y):
        return (preds.argmax(axis=1) == y).mean()

    cbs = [TrainCB(), TqdmCB(), MetricsCB(accuracy=accuracy)]
    learn = Learner(model, dls, loss_func=loss_func, lr=LR, cbs=cbs)
    learn.fit(1)


if __name__ == "__main__":
    main()
