# Copyright 2021-2023 Lawrence Livermore National Security, LLC and other
# MuyGPyS Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: MIT

"""
Noise modeling

Defines data structures and functors that handle noise priors for MuyGPs models.
"""

from MuyGPyS.gp.hyperparameter import ScalarHyperparameter


class NullNoise(ScalarHyperparameter):
    """
    A zero noise assumption model.
    """

    def __init__(self, *args, **kwargs):
        self.val = 0.0
        self.bounds = "fixed"

    def __call__(self, *args, **kwargs):
        return 0.0
