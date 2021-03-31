# Copyright 2021 Lawrence Livermore National Security, LLC and other MuyGPyS 
# Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: (Apache-2.0 OR MIT)

from setuptools import setup

setup(name='MuyGPyS',
      version='0.1',
      description='Scalable Approximate Gaussian Process using Local Kriging',
      author='Benjamin W. Priest',
      author_email='priest2@llnl.gov',
      license='MIT',
      packages=['MuyGPyS'],
      python_requires='>=3.6',
      zip_safe=False)
