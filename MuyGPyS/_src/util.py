# Copyright 2021-2023 Lawrence Livermore National Security, LLC and other
# MuyGPyS Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: MIT

from MuyGPyS import config


def _collect_implementation(package, *funcs):
    if config.state.backend == "numpy":
        return _collect_functions(package + ".numpy", *funcs)
    elif config.state.backend == "jax":
        return _collect_functions(package + ".jax", *funcs)
    elif config.state.backend == "torch":
        return _collect_functions(package + ".torch", *funcs)
    elif config.state.backend == "mpi":
        return _collect_functions(package + ".mpi", *funcs)
    else:
        raise ValueError(
            f'MuyGPyS backend is in bad state "{config.state.backend}"'
        )


def _collect_functions(package, *funcs):
    return tuple([getattr(__import__(package, fromlist=[f]), f) for f in funcs])
