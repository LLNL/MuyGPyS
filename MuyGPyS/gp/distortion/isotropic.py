# Copyright 2021-2023 Lawrence Livermore National Security, LLC and other
# MuyGPyS Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: MIT

import MuyGPyS._src.math as mm
from MuyGPyS._src.gp.tensors import _F2, _l2


class IsotropicDistortion:
    def __init__(self, metric: str):
        if metric == "l2":
            self._dist_fn = _l2
        elif metric == "F2":
            self._dist_fn = _F2
        else:
            raise ValueError(f"Metric {metric} is not supported!")

    def __call__(self, diffs: mm.ndarray) -> mm.ndarray:
        return self._dist_fn(diffs)
