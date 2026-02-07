__all__ = ["DataLoaders", "SimpleDataLoader"]


class SimpleDataLoader:
    def __init__(
        self,
        dataset,
        batch_size,
        shuffle=False,
        drop_last=True,
        transform=None,
    ):
        self.dataset = dataset  # Note it is recommended to use with_format("numpy") for better performance
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.transform = transform

    def __iter__(self):
        dataset = self.dataset
        if self.shuffle:
            dataset = dataset.shuffle()

        i = 0
        while i < len(dataset):
            if self.drop_last and i + self.batch_size > len(dataset):
                break
            batch = dataset[i : i + self.batch_size]
            if self.transform:
                batch = self.transform(batch)
            yield batch
            i += self.batch_size

    def __len__(self):
        full = len(self.dataset) // self.batch_size
        return full if self.drop_last else full + (len(self.dataset) % self.batch_size != 0)


class DataLoaders:
    def __init__(self, *dls):
        self.train, self.valid = dls[:2]

    def get_dls(train_ds, valid_ds, bs, **kwargs):
        return (SimpleDataLoader(train_ds, batch_size=bs, shuffle=True, **kwargs), SimpleDataLoader(valid_ds, batch_size=bs, **kwargs))

    @classmethod
    def from_dd(cls, dd, batch_size, as_tuple=True, **kwargs):
        """
        Create DataLoaders from a dictionary of datasets.
        """
        return cls(*DataLoaders.get_dls(*dd.values(), bs=batch_size, **kwargs))
