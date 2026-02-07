from types import MethodType

from loader import DataLoaders, SimpleDataLoader


def test_simple_dataloader_len_drop_last(xy_dataset_factory):
    dataset = xy_dataset_factory(10)

    drop_last_loader = SimpleDataLoader(dataset, batch_size=4, drop_last=True)
    keep_last_loader = SimpleDataLoader(dataset, batch_size=4, drop_last=False)

    assert len(drop_last_loader) == 2
    assert len(keep_last_loader) == 3


def test_simple_dataloader_iterates_and_transforms(xy_dataset_factory):
    dataset = xy_dataset_factory(4)

    def transform(batch):
        return {
            "x": [value * 10 for value in batch["x"]],
            "y": [value + 1 for value in batch["y"]],
        }

    loader = SimpleDataLoader(dataset, batch_size=2, drop_last=True, transform=transform)
    batches = list(loader)

    assert batches == [
        {"x": [0.0, 10.0], "y": [2.0, 4.0]},
        {"x": [20.0, 30.0], "y": [6.0, 8.0]},
    ]


def test_simple_dataloader_shuffle_uses_dataset_shuffle(xy_dataset_factory):
    dataset = xy_dataset_factory(4)
    calls = {"count": 0}

    def _shuffle(self):
        calls["count"] += 1
        indices = list(reversed(range(len(self))))
        return self.select(indices)

    dataset.shuffle = MethodType(_shuffle, dataset)
    loader = SimpleDataLoader(dataset, batch_size=2, shuffle=True, drop_last=True)

    batches = list(loader)

    assert calls["count"] == 1
    assert batches == [
        {"x": [3.0, 2.0], "y": [7.0, 5.0]},
        {"x": [1.0, 0.0], "y": [3.0, 1.0]},
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
