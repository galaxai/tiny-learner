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

Epoch:0 - Train Loss: 0.699: 100%|████████████████████████████████████████████████| 468/468 [00:35<00:00, 13.14it/s]
MetricsCB - accuracy: 0.6882
Epoch:0 - Valid Loss: 0.729: 100%|██████████████████████████████████████████████████| 78/78 [00:05<00:00, 13.92it/s]
MetricsCB - accuracy: 0.7362
```

## What's included
- `learner.py`: minimal Learner with callback hooks, MetricsCB, TqdmCB, and TinyJit train/valid steps.
- `loader.py`: SimpleDataLoader, DataLoaders, and collate helpers for HF datasets and PIL images.

### Future
- [ ] Fast loader
- [ ] Loader supports num_workers

## License
See `LICENSE`.
