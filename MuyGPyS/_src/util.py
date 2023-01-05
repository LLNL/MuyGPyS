# Copyright 2021-2022 Lawrence Livermore National Security, LLC and other
# MuyGPyS Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: MIT

from MuyGPyS import config


def _collect_implementation(package, *funcs):
    if config.muygpys_backend == "numpy":
        return _collect_functions(package + ".numpy", *funcs)
    elif config.muygpys_backend == "jax":
        return _collect_functions(package + ".jax", *funcs)
    elif config.muygpys_backend == "torch":
        return _collect_functions(package + ".torch", *funcs)
    elif config.muygpys_backend == "mpi":
        return _collect_functions(package + ".mpi", *funcs)
    else:
        raise ValueError(
            f'MuyGPyS backend is in bad state "{config.muygpys_backend}"'
        )


def _collect_functions(package, *funcs):
    return tuple([getattr(__import__(package, fromlist=[f]), f) for f in funcs])
