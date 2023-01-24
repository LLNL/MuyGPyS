# Testing bed for fast precomputation for mean prediction in MuyGPs

This directory contains tests meant specifically for accelerated mean estimation
using precomputation and MuyGPs as described in 
[Fast Gaussian Process Posterior Mean Prediction via Local Cross Validation and Precomputation](https://arxiv.org/abs/2205.10879v1). This feature currently does not support MPI, and can only be run in
JAX, Torch, and Numpy backends.
These tests may be run using the torch backend, which can be turned on using
```
$ export MUYGPYS_BACKEND=torch
``` 
in the user's shell environment. 

If setting environment variables is impractical, one can also use the following
workflow. 

```
from MuyGPyS import config
MuyGPyS.config.update("muygpys_backend","torch")

...subsequent imports from MuyGPyS
```
In order to use JAX, the same commands with "torch" replaced by "jax" may be 
used. 