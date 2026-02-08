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
```bash
>>> uv run examples/train_mnist.py
Epoch:0 - Train Loss: 0.816: 100%|████████████████████████| 468/468 [00:04<00:00, 100.21it/s]
MetricsCB - accuracy: 0.7023
Epoch:0 - Valid Loss: 0.678: 100%|██████████████████████████| 78/78 [00:00<00:00, 132.05it/s]
MetricsCB - accuracy: 0.7532
```

## What's included
- `learner.py`: minimal Learner with callback hooks, MetricsCB, TqdmCB, and TinyJit train/valid steps.
- `loader.py`: SimpleDataLoader and DataLoaders for HF datasets.

### Future
- [x] Fast loader
- [ ] Loader supports num_workers

### Testing
> Tests are written 100% with AI, if you are interested in contributing, please open an pull request.

## License
See `LICENSE`.
