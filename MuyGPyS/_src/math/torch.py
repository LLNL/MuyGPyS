# Copyright 2021-2023 Lawrence Livermore National Security, LLC and other
# MuyGPyS Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: MIT

import torch

from torch import Tensor as _ndarray


def _ones(*args, **kwargs) -> _ndarray:
    return torch.ones(*args, **kwargs)


def _zeros(*args, **kwargs) -> _ndarray:
    return torch.zeros(*args, **kwargs)
