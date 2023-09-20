# Copyright 2021-2023 Lawrence Livermore National Security, LLC and other
# MuyGPyS Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: MIT

"""
Noise modeling base class
"""

from typing import Callable, List, Tuple

import MuyGPyS._src.math as mm


class NoiseFn:
    """
    The basic noise interface
    """

    def __call__(self):
        raise NotImplementedError("base NoiseFn cannot be invoked!")

    def fixed(self) -> bool:
        raise NotImplementedError("base NoiseFn cannot be invoked!")

    def perturb(self, K: mm.ndarray, **kwargs) -> mm.ndarray:
        raise NotImplementedError("base NoiseFn cannot be invoked!")

    def perturb_fn(self, fn: Callable) -> Callable:
        raise NotImplementedError("base NoiseFn cannot be invoked!")

    def append_lists(
        self,
        name: str,
        names: List[str],
        params: List[float],
        bounds: List[Tuple[float, float]],
    ):
        raise NotImplementedError("base NoiseFn cannot be invoked!")
