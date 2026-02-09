__all__ = [
    "CancelFitException",
    "CancelBatchException",
    "CancelEpochException",
    "Callback",
    "run_cbs",
    "SingleBatchCB",
    "TrainCB",
    "TqdmCB",
    "with_cbs",
    "Learner",
    "TrainLearner",
]

from collections.abc import Callable, Sequence
from operator import attrgetter

from tinygrad.engine.jit import TinyJit
from tinygrad.helpers import tqdm
from tinygrad.nn import optim, state
from tinygrad.tensor import Tensor

from loader import DataLoaders


class CancelFitException(Exception):
    pass


class CancelBatchException(Exception):
    pass


class CancelEpochException(Exception):
    pass


CANCEL_EXCEPTIONS = {
    "fit": CancelFitException,
    "batch": CancelBatchException,
    "epoch": CancelEpochException,
}


class Callback:
    order = 0


def run_cbs(cbs: Sequence["Callback"], method_nm: str, learn: "Learner | None" = None) -> None:
    for cb in sorted(cbs, key=attrgetter("order")):
        method = getattr(cb, method_nm, None)
        if method is not None:
            method(learn)


class SingleBatchCB(Callback):
    order = 1

    def after_batch(self, learn):
        raise CancelFitException()


MetricFunc = Callable[[Tensor, Tensor], Tensor]


class MetricsCB(Callback):
    def __init__(self, *ms: MetricFunc, **metrics: MetricFunc) -> None:
        from copy import copy

        for o in ms:
            metrics[type(o).__name__] = o
        self.metrics: dict[str, MetricFunc] = metrics
        self.all_metrics: dict[str, MetricFunc] = copy(metrics)
        self.metric_values: dict[str, list[float]] = {}

    def before_epoch(self, learn: "Learner") -> None:
        self.metric_values = {name: [] for name in self.all_metrics}

    def after_epoch(self, learn: "Learner") -> None:
        parts = []
        for m_name, values in self.metric_values.items():
            if not values:
                continue
            mean_value = sum(values) / len(values)
            parts.append(f"{m_name}: {mean_value:.4f}")
        if parts:
            print(f"{self.__class__.__name__} - {', '.join(parts)}")

    def after_batch(self, learn: "Learner") -> None:
        target = learn.batch[learn.n_inputs]
        for m_name, m_func in self.all_metrics.items():
            value = m_func(learn.preds, target).realize().item()
            self.metric_values[m_name].append(float(value))


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


class TqdmCB(Callback):
    order = MetricsCB.order + 1

    def __init__(self, plot=False):
        self.plot = plot

    def before_epoch(self, learn):
        total = len(learn.dl) if hasattr(learn.dl, "__len__") else None
        learn.dl = tqdm(learn.dl, total=total)

    def after_batch(self, learn):
        if hasattr(learn.dl, "set_description"):
            learn.dl.set_description(f"Epoch:{learn.epoch} - {'Train' if learn.training else 'Valid'} Loss: {learn.loss.item():.3f}")


class with_cbs:
    def __init__(self, nm):
        if nm not in CANCEL_EXCEPTIONS:
            raise ValueError(f"Unknown callback stage: {nm}")
        self.nm = nm
        self.cancel_exc = CANCEL_EXCEPTIONS[nm]

    def __call__(self, f):
        def _f(o, *args, **kwargs):
            try:
                o.callback(f"before_{self.nm}")
                f(o, *args, **kwargs)
                o.callback(f"after_{self.nm}")
            except self.cancel_exc:
                return
            finally:
                o.callback(f"cleanup_{self.nm}")

        return _f


class Learner:
    def __init__(
        self,
        model,
        dls: DataLoaders,
        loss_func: Callable[[Tensor, Tensor], Tensor],
        lr=0.1,
        cbs: Sequence["Callback"] | None = None,
        opt_func=optim.SGD,
        n_inputs=1,
    ):
        self.model = model
        self.dls = dls
        self.loss_func = loss_func
        self.lr = lr
        self.opt_func = opt_func
        self.cbs: list[Callback] = [] if cbs is None else list(cbs)
        self.n_inputs = n_inputs

    @TinyJit
    @Tensor.train()
    def _train_step(self, batch: tuple[Tensor, ...]):
        self.preds = self.model(*batch[: self.n_inputs])
        self.loss = self.loss_func(self.preds, *batch[self.n_inputs :])
        self.loss.backward()
        self.opt.step()
        self.opt.zero_grad()
        return self.loss, self.preds

    @TinyJit
    @Tensor.train(False)
    def _valid_step(self, batch: tuple[Tensor, ...]):
        self.preds = self.model(*batch[: self.n_inputs])
        self.loss = self.loss_func(self.preds, *batch[self.n_inputs :])
        return self.loss, self.preds

    @with_cbs("batch")
    def _one_batch(self):
        # We need to reassign since we are jitting steps
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
