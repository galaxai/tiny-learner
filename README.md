# tiny-learner

Minimal, hackable training loop for tinygrad. It is heavily inspired by the fastai course22p2 learner.

## Install (uv)
Requires Python 3.11+. This project is not on PyPI, so install from GitHub:

```bash
uv pip install "tiny-learner @ git+https://github.com/galaxai/tiny-learner.git"
```

## Quickstart
Clone the repo if you want to run the example:

```bash
git clone https://github.com/galaxai/tiny-learner.git
cd tiny-learner
uv pip install -e .
uv run examples/train_mnist.py
```
```python
>>> uv run examples/train_mnist.py
Epoch:0 - Train Loss: 0.816: 100%|████████████████████████| 468/468 [00:04<00:00, 100.21it/s]
MetricsCB - accuracy: 0.7023
Epoch:0 - Valid Loss: 0.678: 100%|██████████████████████████| 78/78 [00:00<00:00, 132.05it/s]
MetricsCB - accuracy: 0.7532
```
> There is an option to store whole dataset in memory for faster training.
```python
# dls = DataLoaders.from_dd(ds, BATCH_SIZE, transform=transforms, in_memory=True)

Epoch:0 - Train Loss: 0.500: 100%|████████████████████████████████████| 468/468 [00:01<00:00, 330.79it/s]
Epoch:1 - Train Loss: 0.534: 100%|███████████████████████████████████| 468/468 [00:00<00:00, 2731.40it/s]
```

## What's included
- `learner.py`: minimal Learner with callback hooks, MetricsCB, TqdmCB, and TinyJit train/valid steps.
- `loader.py`: SimpleDataLoader and DataLoaders for HF datasets.

### Future
- [ ] Loader supports num_workers
- [ ] Multi-GPU support

### Testing
> Tests are written 100% with AI, if you are interested in fixing this, please open an pull request.

## License
See `LICENSE`.
