[![Develop test](https://github.com/LLNL/MuyGPyS/actions/workflows/develop-test.yml/badge.svg)](https://github.com/LLNL/MuyGPyS/actions/workflows/develop-test.yml)
[![Documentation Status](https://readthedocs.org/projects/muygpys/badge/?version=stable)](https://muygpys.readthedocs.io/en/stable/?badge=stable)
# Fast implementation of the MuyGPs scalable Gaussian process algorithm

MuyGPs is a scalable approximate Gaussian process (GP) model that achieves fast
prediction and model optimization while retaining high-accuracy predictions and
uncertainty quantification.
The MuyGPyS implementation allows the user to easily create GP models that can
quickly train and predict on million-scale problems on a laptop or scale to
billions of observations on distributed memory systems using the same front-end
code.

## What is MuyGPyS?

MuyGPyS is a general-purpose Gaussian process library, similar to
[GPy](https://github.com/SheffieldML/GPy),
[GPyTorch](https://github.com/cornellius-gp/gpytorch), or
[GPflow](https://github.com/GPflow/GPflow).

MuyGPyS differs from the other options in that it constructs approximate GP
models using nearest neighbors sparsification, conditioning predictions only on
the most relevant training data to drastically improve training time and
time-to-solution on large-scale problems.
Indeed, MuyGPyS is intended for GP problems with millions or more observations,
and supports a distributed memory backend for smoothly scaling to billion-scale
problems.

MuyGPs uses nearest neighbors sparsification and performs leave-one-out cross
validation using regularized loss functions to rapidly optimize a GP model without evaluating a much more expensive likelihood, which is required by similar
scalable methods.

## Getting Started 

See the
[illustration tutorial](https://muygpys.readthedocs.io/en/stable/examples/neighborhood_illustration.html)
to see an illustration of why the neighborhood sparsification approach of MuyGPs
works.

Next, see the
[univariate regression tutorial](https://muygpys.readthedocs.io/en/stable/examples/univariate_regression_tutorial.html)
for a full description of the API and an end-to-end walkthrough of a simple
regression problem.

The full documentation, including several additional tutorials with code
examples, can be found at
[readthedocs.io](https://muygpys.readthedocs.io/en/stable/?).

Read further in this document for installation instructions.

## Backend Math Implementation Options

In addition to the default basic numpy backend, as of release v0.6.6, `MuyGPyS`
supports three additional backend implementations of all of its underlying math
functions:

- [MPI](https://github.com/mpi4py/mpi4py) - distributed memory acceleration
- [PyTorch](https://github.com/pytorch/pytorch) - GPU acceleration and neural
network integration
- [JAX](https://github.com/google/jax) - GPU acceleration

It is possible to include the dependencies of any, all, or none of these
additional backends at install time.
Please see the below installation instructions.

`MuyGPyS` uses the `MUYGPYS_BACKEND` environment variable to determine which
backend to use at import time.
It is also possible to manipulate `MuyGPyS.config` to switch between backends
programmatically.
This is not advisable unless the user knows exactly what they are doing
(and must occur before importing any other `MuyGPyS` components).

`MuyGPyS` will default to the `numpy` backend.
It is possible to switch back ends by manipulating the `MUYGPYS_BACKEND`
environment variable in your shell, e.g.
```
$ export MUYGPYS_BACKEND=jax    # turn on JAX backend
$ export MUYGPYS_BACKEND=torch  # turn on Torch backend
$ export MUYGPYS_BACKEND=mpi    # turn on MPI backend
```

### Distributed memory support with MPI

The MPI version of `MuyGPyS` performs all tensor manipulation in distributed
memory.
The tensor creation functions will in fact create and distribute a chunk of each
tensor to each MPI rank.
This data and subsequent data such as posterior means and variances remains
partitioned, and most operations are embarassingly parallel.
Global operations such as loss function computation make use of MPI collectives
like allreduce.
If the user needs to reason about all products of an experiment, such the full
posterior distribution in local memory, it is necessary to employ a collective
such as `MPI.gather`.

The wrapped KNN algorithms are not distributed, and so `MuyGPyS` does not yet
have an internal distributed KNN implementation.
Future versions will support a distributed memory approximate KNN solution.

The user can run a script `myscript.py` with MPI using, e.g. `mpirun` (or `srun`
if using slurm) via
```
$ export MUYGPYS_BACKEND=mpi
$ # mpirun version
$ mpirun -n 4 python myscript.py
$ # srun version
$ srun -N 1 --tasks-per-node 4 -p pbatch python myscript.py
```

### PyTorch Integration

The `torch` version of `MuyGPyS` allows for construction and training of complex
kernels, e.g., convolutional neural network kernels. All low-level math is done
on `torch.Tensor` objects. Due to `PyTorch`'s lack of support for the Bessel 
function of the second kind, we only support special cases of the Matern kernel,
in particular when the smoothness parameter is $\nu = 1/2, 3/2,$ or $5/2$. The
RBF kernel is supported as the Matern kernel with $\nu = \infty$. 

The `MuyGPyS` framework is implemented as a custom `PyTorch` layer. In the 
high-level API found in `examples/muygps_torch`, a `PyTorch` MuyGPs `model` is 
assumed to have two components: a `model.embedding` which deforms the original 
feature data, and a `model.GP_layer` which does Gaussian Process regression on 
the deformed feature space. A code example is provided below.

Most users will want to use the `MuyGPyS.torch.muygps_layer` module to construct 
a custom MuyGPs model. The model can then be calibrated using a standard 
PyTorch training loop. An example of the approach based on the low-level API 
is provided in `docs/examples/torch_tutorial.ipynb`.

In order to use the `MuyGPyS` torch backend, run the following command in your 
shell environment.

```
$ export MUYGPYS_BACKEND=torch
```

One can also use the following workflow to programmatically set the backend to
torch, although the environment variable method is preferred.

```
from MuyGPyS import config
MuyGPyS.config.update("muygpys_backend","torch")

...subsequent imports from MuyGPyS
```

### Just-In-Time Compilation with JAX

`MuyGPyS` supports just-in-time compilation of the
underlying math functions to CPU or GPU using
[JAX](https://github.com/google/jax) since version v0.5.0.
The JAX-compiled versions of the code are significantly faster than numpy,
especially on GPUs.
In order to use the `MuyGPyS` torch backend, run the following command in your
shell environment.

```
$ export MUYGPYS_BACKEND=jax
```

> **_NOTE_**: There is a known conflict between recent versions of `MuyGPyS` and
`JAX` on Python $\geq$ 3.9.
The current fix is to downgrade to Python 3.8.

## Precision

JAX and torch use 32 bit types by default, whereas numpy tends to promote
everything to 64 bits.
For highly stable operations like matrix multiplication, this difference in
precision tends to result in a roughly `1e-8` disagreement between 64 bit and 32
bit implementations.
However, `MuyGPyS` depends upon matrix-vector solves, which can result in
disagreements up to `1e-2`.
Hence, `MuyGPyS` forces all back end implementations to use 64 bit types by
default.

However, the 64 bit operations are slightly slower than their 32 bit
counterparts, and limit throughput on GPUs.
`MuyGPyS` accordingly supports 32 bit types, but this feature is experimental
and might have sharp edges.
For example, `MuyGPyS` might throw errors or otherwise behave strangely if the
user passes arrays of 64 bit types while in 32 bit mode.
Be sure to set your data types appropriately.

A user can have `MuyGPyS`use 32 bit types by setting the `MUYGPYS_FTYPE`
environment variable to `"32"`, e.g.
```
$ export MUYGPYS_FTYPE=32  # use 32 bit types in MuyGPyS functions
```
It is also possible to manipulate `MuyGPyS.config` to switch between types
programmatically.
This is not advisable unless the user knows exactly what they are doing.

## Installation

### Installation using Pip: CPU

The index `muygpys` is maintained on PyPI and can be installed using `pip`.
`muygpys` supports many optional extras flags, which will install additional
dependencies if specified. 
If installing CPU-only with pip, you might want to consider the following flags:  
These extras include:
- `hnswlib` - install [hnswlib](https://github.com/nmslib/hnswlib) dependency to
support fast approximate nearest neighbors indexing
- `jax_cpu` - install [JAX](https://github.com/google/jax) dependencies to 
support just-in-time compilation of math functions on CPU (see below to install
on GPU CUDA architectures)
- `torch` - install [PyTorch](https://github.com/pytorch/pytorch) dependencies
to employ GPU acceleration and the use of the `MuyGPyS.torch` submodule
- `mpi` - install [MPI](https://github.com/mpi4py/mpi4py) dependencies to
support distributed memory parallel computation. Requires that the user has
installed a version of MPI such as
[mvapich](https://mvapich.cse.ohio-state.edu/) or
[open-mpi](https://github.com/open-mpi/ompi).
```
$ # numpy-only installation. Functions will internally use numpy.
$ pip install --upgrade muygpys

$ # The same, but includes hnswlib.
$ pip install --upgrade muygpys[hnswlib]

$ # CPU-only JAX installation. Functions will be jit-compiled using JAX.
$ pip install --upgrade muygpys[jax_cpu]

$ # The same, but includes hnswlib.
$ pip install --upgrade muygpys[jax_cpu,hnswlib]

$ # MPI installation. Functions will operate in distributed memory.
$ pip install --upgrade muygpys[mpi]

$ # The same, but includes hnswlib.
$ pip install --upgrade muygpys[mpi,hnswlib]

$ # pytorch installation. MuyGPyS.torch will be usable.
$ pip install --upgrade muygpys[torch]
```

### Installation using Pip: GPU (CUDA)

#### JAX GPU Instructions

[JAX](https://github.com/google/jax) also supports just-in-time compilation to
CUDA, making the compiled math functions within `MuyGPyS` runnable on NVidia 
GPUS.
This requires you to install 
[CUDA](https://developer.nvidia.com/cuda-downloads) and
[CuDNN](https://developer.nvidia.com/CUDNN)
in your environment, if they are not already installed, and to ensure that they
are on your environment's `$LD_LIBRARY_PATH`. 
See [scripts](scripts/lc-setup/pascal.sh) for an example environment setup.

`MuyGPyS` no longer supports automated GPU-supported JAX installation using pip
extras.
To install JAX as a dependency for `MuyGPyS` to be deployed on cuda-capable
GPUs, please read and follow the
[JAX installation instructions](https://github.com/google/jax#installation).
After installing JAX, the user will also need to install
[Tensorflow Probability](https://github.com/tensorflow/probability) with a JAX
backend via
```
pip install tensorflow-probability[jax]>=0.16.0
```

#### PyTorch GPU Instructions

MuyGPyS does not and most likely will not support installing CUDA PyTorch with
an extras flag.
Please [install PyTorch separately](https://pytorch.org/get-started/locally/).

### Installation From Source

This repository includes several `extras_require` optional dependencies.
- `tests` - install dependencies necessary to run [tests](tests/)
- `docs` - install dependencies necessary to build the docs
- `dev` - install dependencies for maintaining code style, running performance
benchmarks, linting, and packaging

For example, follow these instructions to install from source for development 
purposes with CPU JAX support:
```
$ git clone git@github.com:LLNL/MuyGPyS.git
$ cd MuyGPyS
$ pip install -e .[dev,jax_cpu]
```

If you would like to perform a GPU installation from source, you will need to
install the JAX dependency directly.

Additionally check out the develop branch to access the latest features in 
between stable releases.
See [CONTRIBUTING.md](CONTRIBUTING.md) for contribution rules. 

### Full list of extras flags

- `hnswlib` - install [hnswlib](https://github.com/nmslib/hnswlib) dependency to
support fast approximate nearest neighbors indexing
- `jax_cpu` - install [JAX](https://github.com/google/jax) dependencies to 
support just-in-time compilation of math functions on CPU (see below to install
on GPU CUDA architectures)
- `torch` - install [PyTorch](https://github.com/pytorch/pytorch)
- `mpi` - install [MPI](https://github.com/mpi4py/mpi4py) dependency to support
parallel computation
- `tests` - install dependencies necessary to run [tests](tests/)
- `docs` - install dependencies necessary to build the [docs](docs/)
- `dev` - install dependencies for maintaining code style, linting, and 
packaging

## Building Docs

In order to build the docs locally, first `pip` install from source using either 
the `docs` or `dev` options and then execute:
```
$ sphinx-build -b html docs docs/_build/html
```
Finally, open the file `docs/_build/html/index.html` in your browser of choice.

## Testing

In order to run tests locally, first `pip` install `MuyGPyS` from source using 
the `tests` option.
All tests in the `tests/` directory are then runnable as python scripts, e.g.
```
$ python tests/kernels.py
```

Individual `absl` unit test classes can be run in isolation, e.g.
```
$ python tests/kernels.py DistancesTest
```
It is also possible to run a single method from a test case:
```
$ python tests/kernels.py DistancesTest.test_l2
```

The user can run most tests in all backends.
Some tests use backend-dependent features, and will fail with informative error
messages when attempting an unsupported backend.
The user needs to set `MUYGPYS_BACKEND` and possibly `MUYGPYS_FTYPE` prior to
running the desired test, e.g.,
```
$ export MUYGPYS_BACKEND=jax
$ python tests/kernels.py
```
or
```
$ export MUYGPYS_BACKEND=torch
$ export MUYGPYS_FTYPE=32
$ python tests/backends/torch_correctness.py
```

If the MPI dependencies are installed, the user can also run `absl` tests using
MPI, e.g. using `mpirun`
```
$ export MUYGPYS_BACKEND=mpi
$ mpirun -n 4 python tests/kernels.py
```
or using `srun`
```
$ export MUYGPYS_BACKEND=mpi
$ srun -N 1 --tasks-per-node 4 -p pdebug python tests/kernels.py
```

# About

## Authors

* Benjamin W. Priest (priest2 at llnl dot gov)
* Amanda L. Muyskens (muyskens1 at llnl dot gov)
* Imène Goumiri (goumiri1 at llnl dot gov)

## Papers

MuyGPyS has been used the in the following research papers (newest first):

1. [A Robust Approach to Gaussian Process Implementation](https://arxiv.org/abs/2409.11577)
1. [Enhancing Electrocardiography Data Classification Confidence: A Robust Gaussian Process Approach (MuyGPs)](https://arxiv.org/abs/2409.04642)
1. [Stellar Blend Image Classification Using Computationall Efficient Gaussian Processes](https://arxiv.org/abs/2407.19297)
1. [Closely-Spaced Object Classification Using MuyGPyS](https://arxiv.org/abs/2311.10904)
1. [Light Curve Forecasting and Anomaly Detection Using Scalable, Anisotropic, and Heteroscedastic Gaussian Process Models](https://amostech.com/TechnicalPapers/2023/Poster/Goumiri.pdf)
1. [Scalable Gaussian Process Hyperparameter Optimization via Coverage Regularization](http://export.arxiv.org/abs/2209.11280)
1. [Bayesian Hyperparameter Optimization in Gaussian Processes using Statistical Coverage](https://www.osti.gov/biblio/1902019)
1. [Light Curve Completion and Forecasting Using Fast and Scalable Gaussian Processes (MuyGPs)](https://arxiv.org/abs/2208.14592)
1. [Fast Gaussian Process Posterior Mean Prediction via Local Cross Validation and Precomputation](https://arxiv.org/abs/2205.10879v1)
1. [Gaussian Process Classification of Galaxy Blend Identification in LSST](https://arxiv.org/abs/2107.09246)
1. [MuyGPs: Scalable Gaussian Process Hyperparameter Estimation Using Local Cross-validation](https://arxiv.org/abs/2104.14581)
1. [Star-Galaxy Image Separation with Computationally Efficient Gaussian Process Classification](https://arxiv.org/abs/2105.01106)
1. [Genetic Algorithm for Hyperparameter Optimization in Gaussian Process Modeling](https://www.osti.gov/biblio/1659396)
1. [Star-Galaxy Separation via Gaussian Processes with Model Reduction](https://arxiv.org/abs/2010.06094)

## Citation

If you use MuyGPyS in a research paper, please reference our article:

```
@article{muygps2021,
  title={MuyGPs: Scalable Gaussian Process Hyperparameter Estimation Using Local Cross-Validation},
  author={Muyskens, Amanda and Priest, Benjamin W. and Goumiri, Im{\`e}ne and 
  Schneider, Michael},
  journal={arXiv preprint arXiv:2104.14581},
  year={2021}
}
```

## License

MuyGPyS is distributed under the terms of the MIT license.
All new contributions must be made under the MIT license.

See [LICENSE-MIT](LICENSE-MIT), [NOTICE](NOTICE), and [COPYRIGHT](COPYRIGHT) for 
details.

SPDX-License-Identifier: MIT

## Release

LLNL-CODE-824804
