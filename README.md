# Introduction to variational autoencoders

This project provides an experimentation environment for VAE algorithms.
They can be used for generative modeling and representation learning.
The models are implemented in PyTorch and trained with Lightning.

## Installation

```
pip install -e .
```

## Training

```
python scripts/main.py fit --config config/dense.yaml
```

```
python scripts/main.py fit --config config/conv.yaml
```

## Notebooks

- [Introduction](notebooks/intro.ipynb)

- [Bernoulli VAEs for MNIST](notebooks/binarized_mnist.ipynb)

