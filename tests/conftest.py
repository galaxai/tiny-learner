import pytest
from datasets import Dataset
from tinygrad.tensor import Tensor


def make_xy_data(count):
    xs = [float(i) for i in range(count)]
    ys = [2 * x + 1 for x in xs]
    return {"x": xs, "y": ys}


@pytest.fixture
def xy_data():
    return make_xy_data(6)


@pytest.fixture
def xy_dataset(xy_data):
    return Dataset.from_dict(xy_data)


@pytest.fixture
def xy_dataset_factory():
    def _factory(count):
        return Dataset.from_dict(make_xy_data(count))

    return _factory


@pytest.fixture
def collate_xy():
    def _collate(batch):
        xs = batch["x"]
        ys = batch["y"]
        return Tensor(xs).reshape(len(xs), 1), Tensor(ys).reshape(len(ys), 1)

    return _collate
