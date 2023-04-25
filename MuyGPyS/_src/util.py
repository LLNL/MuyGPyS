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


def _fullname(klass):
    module = klass.__module__
    if module == "builtins":
        return klass.__qualname__
    return module + "." + klass.__qualname__


def _collect_functions(package, *funcs):
    return tuple([getattr(__import__(package, fromlist=[f]), f) for f in funcs])


def auto_str(klass):
    def __str__(self):
        public_members = ", ".join(
            "%s=%s" % item
            for item in vars(self).items()
            if not item[0].startswith("_")
        )
        return f"{type(self).__name__}({public_members})"

    klass.__str__ = __str__
    return klass
