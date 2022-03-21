# Copyright 2021-2022 Lawrence Livermore National Security, LLC and other
# MuyGPyS Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: MIT

try:
    from jax import config as jax_config
    from jax._src.config import Config as JaxConfig
except Exception:
    jax_config = None  # type: ignore


import itertools
import sys


class MuyGPySConfig(JaxConfig):
    def __init__(self):
        super(MuyGPySConfig, self).__init__()
        self.state = MuyGPySState()

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
        self.hnswlib_enabled = False
        self.gpu_enabled = False
        self.already_configured_with_absl = False


config = MuyGPySConfig()


def _update_jax_global(val):
    config.state.jax_enabled = val


def _update_jax_thread_local(val):
    config.state.jax_enabled = val


enable_jax = config.define_bool_state(
    name="muygpys_jax_enabled",
    default=False,
    help="Enable use of jax implementations of math functions.",
    update_global_hook=_update_jax_global,
    update_thread_local_hook=_update_jax_thread_local,
)


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

try:
    import hnswlib as _hnswlib

    config.update("muygpys_hnswlib_enabled", True)
    del _hnswlib
except Exception:
    config.update("muygpys_hnswlib_enabled", False)
