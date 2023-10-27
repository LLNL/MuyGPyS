# Copyright 2021-2023 Lawrence Livermore National Security, LLC and other
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
    "h5py>=3.7.0",
]

DOCS_REQUIRES = [
    "sphinx==6.2.1",
    "sphinx-rtd-theme==1.2.2",
    "sphinx-autodoc-typehints==1.22",
    "matplotlib>=3.2.1",
    "nbsphinx==0.9.2",
    "pandas==1.5.2",
    "pandoc==2.3.0",
    "pandocfilters==1.5.0",
    "ipython==7.30.1",
    "ipykernel==6.6.0",
    "torchvision>=0.14.1",
    "cblind>=2.3.1",
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

MPI_REQUIRES = [
    "mpi4py==3.1.3",
]

TORCH_REQUIRES = [
    "torch>=1.13.0",
]

DOCS_REQUIRES += TORCH_REQUIRES

setup(
    extras_require={
        "dev": DEV_REQUIRES + TEST_REQUIRES + DOCS_REQUIRES,
        "docs": DOCS_REQUIRES,
        "tests": TEST_REQUIRES,
        "hnswlib": HNSWLIB_REQUIRES,
        "jax_cpu": JAX_CPU_REQUIRES + JAX_REQUIRES,
        "mpi": MPI_REQUIRES,
        "torch": TORCH_REQUIRES,
    },
)
