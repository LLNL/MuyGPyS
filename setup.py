# Copyright 2021-2022 Lawrence Livermore National Security, LLC and other
# MuyGPyS Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: MIT

from setuptools import setup


TEST_REQUIRES = [
    "absl-py>=0.13.0",
]

DEV_REQUIRES = [
    "black>=21.1.0",
    "build>=0.7.0",
    "mypy>=0.910",
    "twine>=3.7.1",
]

DOCS_REQUIRES = [
    "sphinx==4.2.0",
    "sphinx-rtd-theme==0.5.2",
    "sphinx-autodoc-typehints==1.12.0",
    "matplotlib>=3.2.1",
    "nbsphinx==0.8.7",
    "pandoc==2.0.1",
    "pandocfilters==1.5.0",
    "ipython==7.30.1",
    "ipykernel==6.6.0",
]

HNSWLIB_REQUIRES = [
    "pybind11>=2.5.0",
    "hnswlib>=0.6.0",
]

JAX_REQUIRES = [
    "tensorflow-probability[jax]>=0.16.0",
]

JAX_CPU_REQUIRES = [
    "jax[cpu]>=0.2.26",
]

JAX_CUDA11_CUDNN805_REQUIRES = [
    "jax[cuda11_cudnn805]",
]

JAX_CUDA11_CUDNN82_REQUIRES = [
    "jax[cuda11_cudnn82]",
]

MPI_REQUIRES = [
    "mpi4py>=3.1.3",
]

setup(
    extras_require={
        "dev": DEV_REQUIRES + TEST_REQUIRES + DOCS_REQUIRES,
        "docs": DOCS_REQUIRES,
        "tests": TEST_REQUIRES,
        "hnswlib": HNSWLIB_REQUIRES,
        "jax_cpu": JAX_CPU_REQUIRES + JAX_REQUIRES,
        "jax_cuda": JAX_CUDA11_CUDNN805_REQUIRES + JAX_REQUIRES,
        "jax_cuda11_cudnn805": JAX_CUDA11_CUDNN805_REQUIRES + JAX_REQUIRES,
        "jax_cuda11_cudnn82": JAX_CUDA11_CUDNN82_REQUIRES + JAX_REQUIRES,
        "mpi": MPI_REQUIRES,
    },
)
