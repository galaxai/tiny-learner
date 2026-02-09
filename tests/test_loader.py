from types import MethodType

from tinygrad.tensor import Tensor

from loader import DataLoaders, SimpleDataLoader


def _batch_values(batch):
    return (batch[0].tolist(), batch[1].tolist())


def _reset_sampler_jit(dl):
    dl.sampler.sample.func.__self__.reset()


def test_simple_dataloader_len_drop_last(xy_dataset_factory):
    dataset = xy_dataset_factory(10)

    drop_last_loader = SimpleDataLoader(dataset, batch_size=4, drop_last=True)
    keep_last_loader = SimpleDataLoader(dataset, batch_size=4, drop_last=False)

    assert len(drop_last_loader) == 2
    assert len(keep_last_loader) == 3


def test_simple_dataloader_iterates_and_transforms(xy_dataset_factory):
    dataset = xy_dataset_factory(4)

    def transform(batch):
        xs = Tensor(batch["x"]) * 10
        ys = Tensor(batch["y"]) + 1
        return xs, ys

    loader = SimpleDataLoader(dataset, batch_size=2, drop_last=True, transform=transform)
    batches = list(loader)

    assert [_batch_values(batch) for batch in batches] == [
        ([0.0, 10.0], [2.0, 4.0]),
        ([20.0, 30.0], [6.0, 8.0]),
    ]


def test_simple_dataloader_in_memory_caches_and_batches(xy_dataset_factory):
    dataset = xy_dataset_factory(128)
    calls = {"count": 0, "size": None}

    def transform(batch):
        calls["count"] += 1
        calls["size"] = len(batch["x"])
        return Tensor(batch["x"]), Tensor(batch["y"])

    loader = SimpleDataLoader(dataset, batch_size=16, in_memory=True, transform=transform)
    _reset_sampler_jit(loader)
    batches_first = list(loader)
    batches_second = list(loader)

    assert len(loader) == 8
    assert calls == {"count": 1, "size": 128}
    assert len(batches_first) == len(batches_second) == len(loader)
    assert _batch_values(batches_first[0]) != _batch_values(batches_first[1])


def test_simple_dataloader_shuffle_uses_dataset_shuffle(xy_dataset_factory):
    dataset = xy_dataset_factory(4)
    calls = {"count": 0}

    def _shuffle(self):
        calls["count"] += 1
        indices = list(reversed(range(len(self))))
        return self.select(indices)

    dataset.shuffle = MethodType(_shuffle, dataset)

    def transform(batch):
        return Tensor(batch["x"]), Tensor(batch["y"])

    loader = SimpleDataLoader(dataset, batch_size=2, shuffle=True, drop_last=True, transform=transform)

    batches = list(loader)

    assert calls["count"] == 1
    assert [_batch_values(batch) for batch in batches] == [
        ([3.0, 2.0], [7.0, 5.0]),
        ([1.0, 0.0], [3.0, 1.0]),
    ]


def test_dataloaders_helpers(xy_dataset_factory):
    train_ds = xy_dataset_factory(4)
    valid_ds = xy_dataset_factory(4)

    train_dl, valid_dl = DataLoaders.get_dls(train_ds, valid_ds, bs=2)
    assert isinstance(train_dl, SimpleDataLoader)
    assert isinstance(valid_dl, SimpleDataLoader)
    assert train_dl.shuffle is True
    assert valid_dl.shuffle is False

    dls = DataLoaders.from_dd({"train": train_ds, "valid": valid_ds}, batch_size=2)
    assert isinstance(dls.train, SimpleDataLoader)
    assert isinstance(dls.valid, SimpleDataLoader)
    assert dls.train.batch_size == 2
    assert dls.valid.batch_size == 2
    assert dls.train.dataset is train_ds
    assert dls.valid.dataset is valid_ds


def test_dataloaders_in_memory_batches_are_randomized(xy_dataset_factory, collate_xy):
    train_ds = xy_dataset_factory(128)
    valid_ds = xy_dataset_factory(128)

    dls = DataLoaders.from_dd({"train": train_ds, "valid": valid_ds}, batch_size=16, in_memory=True, transform=collate_xy)

    _reset_sampler_jit(dls.train)
    train_iter = iter(dls.train)
    train_batch_0 = next(train_iter)
    train_batch_1 = next(train_iter)

    _reset_sampler_jit(dls.valid)
    valid_iter = iter(dls.valid)
    valid_batch_0 = next(valid_iter)
    valid_batch_1 = next(valid_iter)

    assert _batch_values(train_batch_0) != _batch_values(train_batch_1)
    assert _batch_values(valid_batch_0) != _batch_values(valid_batch_1)
