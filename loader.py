__all__ = ["DataLoaders", "SimpleDataLoader"]


from typing import Iterator

from tinygrad.engine.jit import TinyJit
from tinygrad.tensor import Tensor


class InMemorySampler:
    def __init__(self, dl: "SimpleDataLoader"):
        if dl.transform:
            self.data = dl.transform(dl.dataset[:])
        else:
            self.data = dl.dataset[:]
        self.batch_size = dl.batch_size
        self.data_length = len(dl.dataset)

        self.batch_size = dl.batch_size
        self.data_length = len(dl.dataset)

    @TinyJit
    def sample(self, _: int) -> tuple[Tensor, Tensor]:
        samples = Tensor.randint(self.batch_size, high=self.data_length)
        return (self.data[0][samples], self.data[1][samples])


class BatchSampler:
    def __init__(self, dl: "SimpleDataLoader"):
        if dl.shuffle:
            self.dataset = dl.dataset.shuffle()
        else:
            self.dataset = dl.dataset

        self.batch_size = dl.batch_size
        self.data_length = len(dl.dataset)
        self.transform = dl.transform

    def sample(self, i: int) -> tuple[Tensor, Tensor]:
        batch = self.dataset[i : i + self.batch_size]
        if self.transform:
            batch = self.transform(batch)
        return batch


class SimpleDataLoader:
    def __init__(
        self,
        dataset,
        batch_size,
        shuffle=False,
        drop_last=True,
        transform=None,
        in_memory=False,
    ):
        self.dataset = dataset  # Note it is recommended to use with_format("numpy") for better performance
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.transform = transform
        self.in_memory = in_memory

        if self.in_memory:
            self.sampler = InMemorySampler(self)
        else:
            self.sampler = BatchSampler(self)

    def __iter__(self) -> Iterator[tuple[Tensor, Tensor]]:
        i = 0
        while i < len(self.dataset):
            if self.drop_last and i + self.batch_size > len(self.dataset):
                break
            yield self.sampler.sample(i)
            i += self.batch_size

    def __len__(self) -> int:
        full = len(self.dataset) // self.batch_size
        return full if self.drop_last else full + (len(self.dataset) % self.batch_size != 0)


class DataLoaders:
    def __init__(self, *dls):
        self.train, self.valid = dls[:2]

    @staticmethod
    def get_dls(train_ds, valid_ds, bs, **kwargs):
        return (SimpleDataLoader(train_ds, batch_size=bs, shuffle=True, **kwargs), SimpleDataLoader(valid_ds, batch_size=bs, **kwargs))

    @classmethod
    def from_dd(cls, dd, batch_size, **kwargs):
        """
        Create DataLoaders from a dictionary of datasets.
        """
        return cls(*DataLoaders.get_dls(*dd.values(), bs=batch_size, **kwargs))
