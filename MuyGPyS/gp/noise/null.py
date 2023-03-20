# Copyright 2021-2023 Lawrence Livermore National Security, LLC and other
# MuyGPyS Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: MIT

"""
Noise modeling

Defines data structures and functors that handle noise priors for MuyGPs models.
"""


class NullNoise:
    """
    A zero noise assumption model.
    """

    def __call__(self, *args, **kwargs):
        raise NotImplementedError("NullNoise cannot be called!")
