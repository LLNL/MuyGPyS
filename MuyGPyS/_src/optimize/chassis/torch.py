# Copyright 2021-2023 Lawrence Livermore National Security, LLC and other
# MuyGPyS Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: MIT

from typing import Callable

from MuyGPyS.gp import MuyGPS


def _scipy_optimize(
    muygps: MuyGPS,
    obj_fn: Callable,
    verbose: bool = False,
    **kwargs,
) -> MuyGPS:
    raise NotImplementedError(
        "Scipy optimization is not supported in MuyGPs PyTorch module."
    )


def _bayes_opt_optimize(
    muygps: MuyGPS,
    obj_fn: Callable,
    verbose: bool = False,
    **kwargs,
) -> MuyGPS:
    raise NotImplementedError(
        "Bayesian optimization is not supported in MuyGPs PyTorch module."
    )
