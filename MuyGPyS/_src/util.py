# Copyright 2021-2022 Lawrence Livermore National Security, LLC and other
# MuyGPyS Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: MIT

from MuyGPyS import config


def _collect_implementation(package, *funcs):
    if config.muygpys_jax_enabled is False:  # type: ignore
        if config.muygpys_mpi_enabled is False:  # type:ignore
            return _collect_functions(package + ".numpy", *funcs)
        else:
            return _collect_functions(package + ".mpi", *funcs)
    else:
        return _collect_functions(package + ".jax", *funcs)


def _collect_functions(package, *funcs):
    return tuple([getattr(__import__(package, fromlist=[f]), f) for f in funcs])
