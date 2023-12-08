# Introduction to variational autoencoders

## Installation

```
pip install -e .
```

## Training

```
python scripts/train_vae.py --num-features 784 512 128 32 2 --reshape 1 28 28
```

```
python scripts/train_vae.py --num-channels 1 16 24 --num-features 1176 256 32 --reshape 24 7 7
```

## Notebooks

- [Introduction](notebooks/intro.ipynb)

