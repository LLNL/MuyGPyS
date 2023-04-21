# Copyright 2021-2023 Lawrence Livermore National Security, LLC and other
# MuyGPyS Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: MIT

try:
    from jax import config as jax_config
    from jax._src.config import Config as JaxConfig
except Exception:
    jax_config = None  # type: ignore
    from MuyGPyS._src.jaxconfig import Config as JaxConfig  # type: ignore

import itertools
import sys


class MuyGPySConfig(JaxConfig):
    def __init__(self):
        super(MuyGPySConfig, self).__init__()
        self.state = MuyGPySState()
        self.mpi_state = MPIState()

    def parse_flags_with_absl(self):
        if self.state.already_configured_with_absl is False:
            # Extract just the --muygpys... flags (before the first --) from
            # argv. In some environments (e.g. ipython/colab) argv might be a
            # mess of things parseable by absl and other junk.
            muygpys_argv = itertools.takewhile(lambda a: a != "--", sys.argv)
            muygpys_argv = [
                "",
                *(a for a in muygpys_argv if a.startswith("--muygpys")),
            ]

            import absl.flags

            self.config_with_absl()
            absl.flags.FLAGS(muygpys_argv, known_only=True)
            self.complete_absl_config(absl.flags)
            self.state.already_configured_with_absl = True


class MuyGPySState:
    def __init__(self):
        self.jax_enabled = False
        self.torch_enabled = False
        self.hnswlib_enabled = False
        self.gpu_enabled = False
        self.mpi_enabled = False
        self.backend = "numpy"
        self.ftype = "64"
        self.already_configured_with_absl = False

    def low_precision(self):
        return True if self.ftype == "32" else False


class MPIState:
    def __init__(self):
        self._comm_world = None

    @property
    def comm_world(self):
        return self._comm_world

    def set_comm(self, comm):
        self._comm_world = comm


config = MuyGPySConfig()


def _update_jax_global(val):
    config.state.jax_enabled = val


def _update_jax_thread_local(val):
    config.state.jax_enabled = val


# JAX and GPU states

enable_jax = config.define_bool_state(
    name="muygpys_jax_enabled",
    default=False,
    help="Enable use of jax implementations of math functions.",
    update_global_hook=_update_jax_global,
    update_thread_local_hook=_update_jax_thread_local,
)


def _update_gpu_global(val):
    config.state.gpu_enabled = val


def _update_gpu_thread_local(val):
    config.state.gpu_enabled = val


enable_gpu = config.define_bool_state(
    name="muygpys_gpu_enabled",
    default=False,
    help="Enable use of GPUs with JAX.",
    update_global_hook=_update_gpu_global,
    update_thread_local_hook=_update_gpu_thread_local,
)


try:
    from jax import default_backend as _default_backend

    config.update("muygpys_jax_enabled", True)
    if _default_backend() in ["gpu", "tpu"]:
        config.update("muygpys_gpu_enabled", True)
    del _default_backend
except Exception:
    config.update("muygpys_jax_enabled", False)
    config.update("muygpys_gpu_enabled", False)


# hnswlib states


def _update_hnswlib_global(val):
    config.state.hnswlib_enabled = val


def _update_hnswlib_thread_local(val):
    config.state.hnswlib_enabled = val


enable_hnswlib = config.define_bool_state(
    name="muygpys_hnswlib_enabled",
    default=False,
    help="Enable use of hnswlib implementation of fast approximate nearest neighbors.",
    update_global_hook=_update_hnswlib_global,
    update_thread_local_hook=_update_hnswlib_thread_local,
)


try:
    import hnswlib as _hnswlib

    config.update("muygpys_hnswlib_enabled", True)
    del _hnswlib
except Exception:
    config.update("muygpys_hnswlib_enabled", False)


# MPI states


def _update_mpi_global(val):
    config.state.mpi_enabled = val


def _update_mpi_thread_local(val):
    config.state.mpi_enabled = val


enable_mpi = config.define_bool_state(
    name="muygpys_mpi_enabled",
    default=False,
    help="Enable use of mpi for parallelization.",
    update_global_hook=_update_mpi_global,
    update_thread_local_hook=_update_mpi_thread_local,
)

try:
    from mpi4py import MPI
    from mpi4py.util.pkl5 import Intracomm

    # wrap COMM_WORLD with pkl5 for large number of messages per
    # https://mpi4py.readthedocs.io/en/3.1.4/mpi4py.util.pkl5.html
    config.mpi_state.set_comm(Intracomm(MPI.COMM_WORLD))

    config.update("muygpys_mpi_enabled", True)
except Exception:
    MPI = None  # type: ignore
    config.update("muygpys_mpi_enabled", False)


# torch states


def _update_torch_global(val):
    config.state.torch_enabled = val


def _update_torch_thread_local(val):
    config.state.torch_enabled = val


enable_torch = config.define_bool_state(
    name="muygpys_torch_enabled",
    default=False,
    help="Enable use of torch for deep kernel learning.",
    update_global_hook=_update_torch_global,
    update_thread_local_hook=_update_torch_thread_local,
)


try:
    import torch as _torch

    config.update("muygpys_torch_enabled", True)
    del _torch
except Exception:
    config.update("muygpys_torch_enabled", False)

# Backend state


def _update_backend_global(val):
    config.state.backend = val


def _update_backend_thread_local(val):
    config.state.backend = val


backend = config.define_enum_state(
    name="muygpys_backend",
    enum_values=["numpy", "jax", "torch", "mpi"],
    default="numpy",
    help="Specify which backend to select at import time",
    update_global_hook=_update_backend_global,
    update_thread_local_hook=_update_backend_thread_local,
)

if (
    config.state.backend == "jax"
    and config.state.jax_enabled is False
    or config.state.backend == "torch"
    and config.state.torch_enabled is False
    or config.state.backend == "mpi"
    and config.state.mpi_enabled is False
):
    be = config.state.backend
    raise ValueError(
        f'MuyGPyS backend is set to "{be}", but "{be}" is not enabled! '
        f"The {be} dependencies are most likely not installed in your "
        f"environment."
    )


def _update_ftype_global(val):
    config.state.ftype = val


def _update_ftype_thread_local(val):
    config.state.ftype = val


ftype = config.define_enum_state(
    name="muygpys_ftype",
    enum_values=["32", "64"],
    default="64",
    help="Specify the float precision to be used",
    update_global_hook=_update_ftype_global,
    update_thread_local_hook=_update_ftype_thread_local,
)

if config.state.jax_enabled is True and config.state.ftype == "64":
    jax_config.update("jax_enable_x64", True)
