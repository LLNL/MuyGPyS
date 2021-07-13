# Copyright 2021 Lawrence Livermore National Security, LLC and other MuyGPyS 
# Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: (Apache-2.0 OR MIT)

from setuptools import setup

setup(name='MuyGPyS',
      version='0.2.1',
      description='Scalable Approximate Gaussian Process using Local Kriging',
      author='Benjamin W. Priest',
      author_email='priest2@llnl.gov',
      license='MIT',
      packages=['MuyGPyS'],
      python_requires='>=3.6',
      install_requires=[
          'numpy',
          'scipy==1.4.1',
          'scikit-learn',
          'absl-py',
          'pybind11',
          'hnswlib',
      ],
      zip_safe=False)
