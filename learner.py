__all__ = [
    "CancelFitException",
    "CancelBatchException",
    "CancelEpochException",
    "Callback",
    "run_cbs",
    "SingleBatchCB",
    "TrainCB",
    "tqdmCB",
    "with_cbs",
    "Learner",
    "TrainLearner",
]

from collections.abc import Sequence
from operator import attrgetter
from typing import Callable

from tinygrad import Tensor
from tinygrad.engine.jit import TinyJit
from tinygrad.helpers import tqdm
from tinygrad.nn import optim, state


class CancelFitException(Exception):
    pass


class CancelBatchException(Exception):
    pass


class CancelEpochException(Exception):
    pass


class Callback:
    order = 0


def run_cbs(cbs, method_nm, learn=None):
    for cb in sorted(cbs, key=attrgetter("order")):
        method = getattr(cb, method_nm, None)
        if method is not None:
            method(learn)


class SingleBatchCB(Callback):
    order = 1

    def after_batch(self, learn):
        raise CancelFitException()


# def_device = Tensor([0]).device


# def to_device(x, device=def_device):
#     if isinstance(x, Tensor):
#         return x.to(device)
#     if isinstance(x, Mapping):
#         return {k: to_device(v, device) for k, v in x.items()}
#     if isinstance(x, list):
#         return [to_device(o, device) for o in x]
#     if isinstance(x, tuple):
#         return tuple(to_device(o, device) for o in x)
#     return x


# class Mean:
#     def __init__(self):
#         self.reset()

#     def reset(self):
#         self.total = 0.0
#         self.count = 0.0

#     def update(self, value, weight=1.0):
#         self.total += _to_float(value) * weight
#         self.count += weight

#     def compute(self):
#         return self.total / self.count if self.count else math.nan


# class MulticlassAccuracy:
#     def __init__(self):
#         self.reset()

#     def reset(self):
#         self.correct = 0
#         self.total = 0

#     def update(self, preds, target):
#         p = _to_numpy(preds)
#         t = _to_numpy(target)
#         if p.ndim > 1:
#             p = p.argmax(axis=-1)
#         if t.ndim > 1:
#             t = t.argmax(axis=-1)
#         p = p.reshape(-1)
#         t = t.reshape(-1)
#         self.correct += int((p == t).sum())
#         self.total += int(t.size)

#     def compute(self):
#         return self.correct / self.total if self.total else math.nan


# class MetricsCB(Callback):
#     def __init__(self, *ms, **metrics):
#         for o in ms:
#             metrics[type(o).__name__] = o
#         self.metrics = metrics
#         self.all_metrics = copy(metrics)
#         self.all_metrics["loss"] = self.loss = Mean()

#     def _log(self, d):
#         print(d)

#     def before_fit(self, learn):
#         learn.metrics = self

#     def before_epoch(self, learn):
#         [o.reset() for o in self.all_metrics.values()]

#     def after_epoch(self, learn):
#         log = {k: f"{v.compute():.3f}" for k, v in self.all_metrics.items()}
#         log["epoch"] = learn.epoch
#         log["train"] = "train" if learn.training else "eval"
#         self._log(log)

#     def after_batch(self, learn):
#         x, y, *_ = to_cpu(learn.batch)
#         for m in self.metrics.values():
#             m.update(to_cpu(learn.preds), y)
#         self.loss.update(to_cpu(learn.loss), weight=len(x))


# class DeviceCB(Callback):
#     def __init__(self, device=def_device):
#         self.device = device

#     def before_batch(self, learn):
#         learn.batch = to_device(learn.batch, device=self.device)


class TrainCB(Callback):
    def __init__(self, n_inputs=1):
        self.n_inputs = n_inputs

    def predict(self, learn):
        learn.preds = learn.model(*learn.batch[: self.n_inputs])

    def get_loss(self, learn):
        learn.loss = learn.loss_func(learn.preds, *learn.batch[self.n_inputs :])

    def backward(self, learn):
        learn.loss.backward()

    def step(self, learn):
        learn.opt.step()

    def zero_grad(self, learn):
        learn.opt.zero_grad()


class tqdmCB(Callback):
    # order = MetricsCB.order + 1

    def __init__(self, plot=False):
        self.plot = plot

    def before_epoch(self, learn):
        total = len(learn.dl) if hasattr(learn.dl, "__len__") else None
        learn.dl = tqdm(learn.dl, desc="", total=total)

    def after_batch(self, learn):
        if hasattr(learn.dl, "set_description"):
            learn.dl.set_description(f"Epoch:{learn.epoch} - {'Train' if learn.training else 'Valid'} Loss: {learn.loss.item():.3f}")


class with_cbs:
    def __init__(self, nm):
        self.nm = nm

    def __call__(self, f):
        def _f(o, *args, **kwargs):
            try:
                o.callback(f"before_{self.nm}")
                f(o, *args, **kwargs)
                o.callback(f"after_{self.nm}")
            except globals()[f"Cancel{self.nm.title()}Exception"]:
                pass
            finally:
                o.callback(f"cleanup_{self.nm}")

        return _f


class Learner:
    def __init__(
        self,
        model,
        dls,
        loss_func: Callable[[Tensor, Tensor], Tensor],
        lr=0.1,
        cbs: Sequence[Callable[[str], None]] | None = None,
        opt_func=optim.SGD,
        n_inputs=1,
    ):
        self.model = model
        self.dls = dls
        self.loss_func = loss_func
        self.lr = lr
        self.opt_func = opt_func
        self.cbs = [] if cbs is None else list(cbs)
        self.n_inputs = n_inputs

    @TinyJit
    @Tensor.train()
    def _train_step(self, batch):
        self.preds = self.model(*batch[: self.n_inputs])
        self.loss = self.loss_func(self.preds, *batch[self.n_inputs :])
        self.loss.backward()
        self.opt.step()
        self.opt.zero_grad()
        return self.loss, self.preds

    @TinyJit
    @Tensor.train(False)
    def _valid_step(self, batch):
        self.preds = self.model(*batch[: self.n_inputs])
        self.loss = self.loss_func(self.preds, *batch[self.n_inputs :])
        return self.loss, self.preds

    @with_cbs("batch")
    def _one_batch(self):
        # We need to reasign to self since we are jitting steps
        if self.training:
            self.loss, self.preds = self._train_step(self.batch)
        else:
            self.loss, self.preds = self._valid_step(self.batch)

    @with_cbs("epoch")
    def _one_epoch(self):
        for self.iter, self.batch in enumerate(self.dl):
            self._one_batch()

    def one_epoch(self, training):
        self.dl = self.dls.train if training else self.dls.valid
        with Tensor.train(training):
            self._one_epoch()

    @with_cbs("fit")
    def _fit(self, train, valid):
        for self.epoch in self.epochs:
            if train:
                self.one_epoch(True)
            if valid:
                self.one_epoch(False)

    def fit(self, n_epochs=1, train=True, valid=True):
        self.n_epochs = n_epochs
        self.epochs = range(n_epochs)
        if self.opt_func:
            self.opt = self.opt_func(state.get_parameters(self.model), lr=self.lr)
        self._fit(train, valid)

    def callback(self, method_nm):
        run_cbs(self.cbs, method_nm, self)

    @property
    def training(self):
        return Tensor.training


class TrainLearner(Learner):
    def predict(self):
        self.preds = self.model(self.batch[0])

    def get_loss(self):
        self.loss = self.loss_func(self.preds, self.batch[1])

    def backward(self):
        self.loss.backward()

    def step(self):
        self.opt.step()

    def zero_grad(self):
        self.opt.zero_grad()
