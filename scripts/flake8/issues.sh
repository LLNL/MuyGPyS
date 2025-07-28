#!/usr/bin/env bash

# Copyright 2021-2023 Lawrence Livermore National Security, LLC and other
# MuyGPyS Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: MIT

# usage:
# $ cd ${MUYGPYS_ROOT}
# $ sh ./scripts/flake_lint

flake8 . --count --max-complexity=10 --max-line-length=127 --statistics --ignore=W503,E203 --per-file-ignores="__init__.py:F401 jax.py:F401 mpi.py:F401,F403 numpy.py:F401 torch.py:F401 shear.py:E402"
