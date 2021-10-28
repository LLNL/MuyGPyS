# Copyright 2021 Lawrence Livermore National Security, LLC and other MuyGPyS
# Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: MIT

import os
import sys

from setuptools import find_packages, setup
from typing import Dict


this_directory = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(this_directory, "README.md"), encoding="utf-8") as f:
    long_description = f.read()

INSTALL_REQUIRES = [
    "numpy>=1.18.5",
    "scipy==1.4.1",
    "scikit-learn>=0.23.2",
    "absl-py>=0.13.0",
    "pybind11>=2.5.0",
    "hnswlib>=0.5.2",
]

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
    name="muygpys",
    version=_get_version(),
    description="Scalable Approximate Gaussian Process using Local Kriging",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Benjamin W. Priest",
    author_email="priest2@llnl.gov",
    license="MIT",
    packages=find_packages(),
    python_requires=">=3.6",
    install_requires=INSTALL_REQUIRES,
    url="https://github.com/LLNL/MuyGPyS",
    download_url="https://pypi.org/project/muygpys",
    project_urls={
        "Source Code": "https://github.com/LLNL/MuyGPyS",
        "Documentation": "https://muygpys.readthedocs.io",
        "Bug Tracker": "https://github.com/LLNL/MuyGPyS/issues",
    },
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
