# Max-product noisy-OR

This repo contains code for the paper [Learning noisy-OR Bayesian Networks with Max-Product Belief Propagation](https://arxiv.org/abs/2302.00099) accepted at the International Conference on Machine Learning 2023.

## Installation

### Install from GitHub
```
pip install git+https://github.com/deepmind/max_product_noisy_or.git
```

### Developer
You need Python 3.10 to get started.
While you can install this package in your standard python environment,
we recommend using a
[Python virtual environment](https://docs.python.org/3/tutorial/venv.html)
to manage your dependencies. This should help to avoid version conflicts and
just generally make the installation process easier.

```
git clone https://github.com/deepmind/max_product_noisy_or.git
cd max_product_noisy_or
python -m venv mp_noisy_or_env
source mp_noisy_or_env/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -e .
python setup.py install
```

### Install on GPU

By default the above commands install JAX for CPU. If you have access to a GPU,
follow the official instructions [here](https://github.com/google/jax#pip-installation-gpu-cuda)
to install JAX for GPU.

## Getting Started

The [example script](https://github.com/deepmind/max_product_noisy_or/blob/main/examples/example.py) shows a training and testing demonstration and can be run via
```
python3 examples/example.py
```
Its [notebook version](https://colab.research.google.com/github/deepmind/max_product_noisy_or/blob/master/examples/example.ipynb) displays some additional figures.


## Note

This is not an officially supported Google product.
