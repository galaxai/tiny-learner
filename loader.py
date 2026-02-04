__all__ = ["DataLoaders", "SimpleDataLoader", "collate_dict", "default_collate", "pil_to_tensor"]

import random
from collections.abc import Mapping
from operator import itemgetter

from PIL.Image import Image
from tinygrad import Tensor
from tinygrad.dtype import dtypes


def pil_to_tensor(img: Image, pixel_format="RGB"):
    """
    Return HWC Tensor from PIL Image
    """
    ##https://github.com/tinygrad/tinygrad/blob/master/tinygrad/nn/onnx.py#L586
    height, width = img.height, img.width
    if pixel_format == "BGR":
        return Tensor(img.tobytes(), dtype=dtypes.uint8).reshape(height, width, 3).flip(-1)
    if pixel_format == "RGB":
        return Tensor(img.tobytes(), dtype=dtypes.uint8).reshape(height, width, 3)
    if pixel_format == "Grayscale":
        return Tensor(img.convert("L").tobytes(), dtype=dtypes.uint8).reshape(height, width, 1)
    raise ValueError(f"pixel_format={pixel_format!r} is not supported.")


class SimpleDataLoader:
    def __init__(
        self,
        dataset,
        batch_size,
        shuffle=False,
        drop_last=True,
        transform=None,
        collate_fn=None,
    ):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.transform = transform
        self.collate_fn = collate_fn or default_collate

    def __iter__(self):
        idxs = list(range(len(self.dataset)))
        if self.shuffle:
            random.shuffle(idxs)
        bs = self.batch_size
        for start in range(0, len(idxs), bs):
            batch_idxs = idxs[start : start + bs]
            if self.drop_last and len(batch_idxs) < bs:
                break
            samples = [self.dataset[i] for i in batch_idxs]
            if self.transform:
                samples = [self.transform(s) for s in samples]
            yield self.collate_fn(samples)

    def __len__(self):
        full = len(self.dataset) // self.batch_size
        return full if self.drop_last else full + (len(self.dataset) % self.batch_size != 0)


def default_collate(batch):
    if len(batch) == 0:
        return batch
    first = batch[0]
    if isinstance(first, Mapping):
        return {k: default_collate([b[k] for b in batch]) for k in first}
    if isinstance(first, tuple):
        return tuple(default_collate(list(samples)) for samples in zip(*batch))
    if isinstance(first, list):
        return [default_collate(list(samples)) for samples in zip(*batch)]
    if isinstance(first, Tensor):
        return Tensor.stack(batch)
    if hasattr(first, "shape") and not isinstance(first, (str, bytes)):
        return Tensor.stack([Tensor(o) for o in batch])
    if isinstance(first, (int, float, bool)):
        return Tensor(batch)
    return batch


def collate_dict(ds):
    get = itemgetter(*ds.features)

    def _f(b):
        return get(default_collate(b))

    return _f


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
        f = collate_dict(dd["train"])
        return cls(*DataLoaders.get_dls(*dd.values(), bs=batch_size, collate_fn=f, **kwargs))
