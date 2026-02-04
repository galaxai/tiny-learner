from datasets import load_dataset
from tinygrad import Tensor, nn
from tinygrad.engine.jit import TinyJit

from learner import Learner, MetricsCB, TqdmCB, TrainCB
from loader import DataLoaders, pil_to_tensor


class TinyMLP:
    def __init__(self, in_features=28 * 28, hidden=128, out_features=10):
        self.l1 = nn.Linear(in_features, hidden)
        self.l2 = nn.Linear(hidden, out_features)

    def __call__(self, x):
        return self.l2(self.l1(x).relu())


def transforms(s: Tensor):
    x = "image"
    s[x] = [pil_to_tensor(o, pixel_format="Grayscale").flatten() for o in s[x]]
    return s


def loss_func(preds, y):
    return preds.cross_entropy(y)


BATCH_SIZE = 128
LR = 1e-3


def main():
    ds = load_dataset("zalando-datasets/fashion_mnist")
    tds = ds.with_transform(transforms)

    dls = DataLoaders.from_dd(tds, BATCH_SIZE)
    model = TinyMLP()

    @TinyJit
    def accuracy(preds, y):
        return (preds.argmax(axis=1) == y).mean()

    cbs = [TrainCB(), TqdmCB(), MetricsCB(accuracy=accuracy)]
    learn = Learner(model, dls, loss_func=loss_func, lr=LR, cbs=cbs)
    learn.fit(1)


if __name__ == "__main__":
    main()
