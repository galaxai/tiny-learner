__all__ = ["DataLoaders", "SimpleDataLoader"]


from typing import Iterator

from tinygrad.engine.jit import TinyJit
from tinygrad.tensor import Tensor


class Sampler:
    """Base sampler that yields tensor batches from a data loader."""

    def __init__(self, dl: "SimpleDataLoader"):
        self.data_len = len(dl.dataset)
        self.batch_size = dl.batch_size
        self.drop_last = dl.drop_last

    def __iter__(self) -> Iterator[tuple[Tensor, ...]]:
        raise NotImplementedError("Sampler.__iter__ must be implemented")

    def __len__(self) -> int:
        full = self.data_len // self.batch_size
        return full if self.drop_last else full + (self.data_len % self.batch_size != 0)


class InMemorySampler(Sampler):
    """Randomly sample batches from an in-memory dataset"""

    def __init__(self, dl: "SimpleDataLoader"):
        super().__init__(dl)
        if dl.transform:
            self.data = dl.transform(dl.dataset[:])
        else:
            self.data = dl.dataset[:]
        self.batch_size = dl.batch_size

    @TinyJit
    def sample(self, _: int) -> tuple[Tensor, ...]:
        # This will miss some data samples due to the randomness
        samples = Tensor.randint(self.batch_size, high=self.data_len)
        return tuple(col[samples] for col in self.data)

    def __iter__(self) -> Iterator[tuple[Tensor, ...]]:
        for i in range(0, self.__len__()):
            yield self.sample(i)


class BatchSampler(Sampler):
    """Yield sequential batches with optional shuffle and transform."""

    def __init__(self, dl: "SimpleDataLoader"):
        super().__init__(dl)
        self.shuffle = dl.shuffle
        self.dataset = dl.dataset
        self.transform = dl.transform

    def __iter__(self):
        self.data = self.dataset
        if self.shuffle:
            self.data = self.dataset.shuffle()

        for i in range(0, self.__len__()):
            i *= self.batch_size
            batch = self.data[i : i + self.batch_size]
            if self.transform:
                batch = self.transform(batch)
            yield batch


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

    def __iter__(self) -> Iterator[tuple[Tensor, ...]]:
        for batch in iter(self.sampler):
            yield batch

    def __len__(self) -> int:
        return self.sampler.__len__()


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
