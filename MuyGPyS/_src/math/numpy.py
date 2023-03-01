# Copyright 2021-2023 Lawrence Livermore National Security, LLC and other
# MuyGPyS Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: MIT

import numpy as np


def _ones(*args, **kwargs) -> np.ndarray:
    return np.ones(*args, **kwargs)


def _zeros(*args, **kwargs) -> np.ndarray:
    return np.zeros(*args, **kwargs)
