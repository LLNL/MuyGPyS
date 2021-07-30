# Copyright 2021 Lawrence Livermore National Security, LLC and other MuyGPyS
# Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: (Apache-2.0 OR MIT)

import os
import sys

from setuptools import setup
from typing import Dict


def _get_version() -> str:
    """Returns the package version.

    Adapted from:
    https://github.com/deepmind/dm-haiku/blob/d4807e77b0b03c41467e24a247bed9d1897d336c/setup.py#L22

    Returns:
      Version number.
    """
    path = "MuyGPyS/__init__.py"
    version = "__version__"
    with open(path) as fp:
        for line in fp:
            if line.startswith(version):
                g: Dict = {}
                exec(line, g)  # pylint: disable=exec-used
                return g[version]  # pytype: disable=key-error
        raise ValueError(f"`{version}` not defined in `{path}`.")


setup(
    name="MuyGPyS",
    version=_get_version(),
    description="Scalable Approximate Gaussian Process using Local Kriging",
    author="Benjamin W. Priest",
    author_email="priest2@llnl.gov",
    license="MIT",
    packages=["MuyGPyS"],
    python_requires=">=3.6",
    project_urls={
        "Source Code": "https://github.com/LLNL/MuyGPyS",
        "Documentation": "https://muygpys.readthedocs.io",
        "Bug Tracker": "https://github.com/LLNL/MuyGPyS/issues",
    },
    install_requires=[
        "numpy",
        "scipy==1.4.1",
        "scikit-learn",
        "absl-py",
        "pybind11",
        "hnswlib",
    ],
    classifiers=[
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "License :: OSI Approved :: MIT License",
        "Operating System :: MacOS",
        "Operating System :: POSIX :: Linux",
        "Topic :: Software Development",
        "Topic :: Software Development :: Libraries",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Mathematics",
        "Topic :: Scientific/Engineering",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Development Status :: 3 - Alpha",
    ],
    zip_safe=False,
)
