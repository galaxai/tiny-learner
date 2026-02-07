import pytest
from tinygrad.tensor import Tensor

from learner import (
    Callback,
    CancelBatchException,
    Learner,
    MetricsCB,
    SingleBatchCB,
    run_cbs,
    with_cbs,
)
from loader import DataLoaders, SimpleDataLoader


class LinearModel:
    def __init__(self):
        self.w = Tensor([0.0], requires_grad=True)
        self.b = Tensor([0.0], requires_grad=True)

    def __call__(self, x):
        return x * self.w + self.b


def mse(preds, target):
    diff = preds - target
    return (diff * diff).mean()


def build_learner(dataset, batch_size, collate, cbs=None, lr=0.1):
    dataset = dataset.with_format("numpy")
    dl = SimpleDataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=True, transform=collate)
    dls = DataLoaders(dl, dl)
    model = LinearModel()
    learner = Learner(model, dls, loss_func=mse, lr=lr, cbs=cbs)
    return learner, model, dl


def test_run_cbs_orders_callbacks():
    calls = []

    class FirstCB(Callback):
        order = 1

        def log(self, learn):
            calls.append("first")

    class SecondCB(Callback):
        order = 0

        def log(self, learn):
            calls.append("second")

    run_cbs([FirstCB(), SecondCB()], "log")
    assert calls == ["second", "first"]


def test_with_cbs_calls_before_after_cleanup():
    calls = []

    class LogCB(Callback):
        def before_batch(self, learn):
            calls.append("before")

        def after_batch(self, learn):
            calls.append("after")

        def cleanup_batch(self, learn):
            calls.append("cleanup")

    class Runner:
        def __init__(self):
            self.cbs = [LogCB()]

        def callback(self, method_nm):
            run_cbs(self.cbs, method_nm)

        @with_cbs("batch")
        def run(self):
            calls.append("run")

    Runner().run()
    assert calls == ["before", "run", "after", "cleanup"]


def test_with_cbs_cancel_runs_cleanup():
    calls = []

    class CancelCB(Callback):
        def before_batch(self, learn):
            calls.append("before")
            raise CancelBatchException()

        def cleanup_batch(self, learn):
            calls.append("cleanup")

    class Runner:
        def __init__(self):
            self.cbs = [CancelCB()]

        def callback(self, method_nm):
            run_cbs(self.cbs, method_nm)

        @with_cbs("batch")
        def run(self):
            calls.append("run")

    Runner().run()
    assert calls == ["before", "cleanup"]


def test_with_cbs_unknown_stage_raises():
    with pytest.raises(ValueError):
        with_cbs("unknown")


def test_learner_fit_updates_params(xy_dataset_factory, collate_xy):
    dataset = xy_dataset_factory(4)
    learner, model, _ = build_learner(dataset, batch_size=4, collate=collate_xy)

    w_before = float(model.w.item())
    learner.fit(n_epochs=1, train=True, valid=False)
    w_after = float(model.w.item())

    assert abs(w_after - w_before) > 1e-6


def test_learner_valid_does_not_update_params(xy_dataset_factory, collate_xy):
    dataset = xy_dataset_factory(4)
    learner, model, _ = build_learner(dataset, batch_size=4, collate=collate_xy)

    w_before = float(model.w.item())
    learner.fit(n_epochs=1, train=False, valid=True)
    w_after = float(model.w.item())

    assert w_after == pytest.approx(w_before)


def test_single_batch_cb_stops_fit(xy_dataset_factory, collate_xy):
    dataset = xy_dataset_factory(8)

    class CountCB(Callback):
        def __init__(self):
            self.count = 0

        def after_batch(self, learn):
            self.count += 1

    count_cb = CountCB()
    learner, _, _ = build_learner(dataset, batch_size=2, collate=collate_xy, cbs=[count_cb, SingleBatchCB()])
    learner.fit(n_epochs=1, train=True, valid=False)

    assert count_cb.count == 1


def test_metrics_cb_records_values(xy_dataset_factory, collate_xy):
    dataset = xy_dataset_factory(6)
    metrics = MetricsCB(mse=mse)
    learner, _, dl = build_learner(dataset, batch_size=2, collate=collate_xy, cbs=[metrics])

    learner.fit(n_epochs=1, train=True, valid=False)

    assert len(metrics.metric_values["mse"]) == len(dl)
    assert all(isinstance(value, float) for value in metrics.metric_values["mse"])
