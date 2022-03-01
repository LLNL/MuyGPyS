# Copyright 2021 Lawrence Livermore National Security, LLC and other MuyGPyS
# Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: MIT

try:
    from jax import config as jax_config
except Exception:
    jax_config = None  # type: ignore


class Config:
    _HAS_DYNAMIC_ATTRIBUTES = True

    def __init__(self):
        self._jax_enabled = False
        self._gpu_found = False
        try:
            from jax import default_backend as _default_backend

            self._jax_enabled = True
            if _default_backend() in ["gpu", "tpu"]:
                self._gpu_found = True
            del _default_backend

        except Exception:
            self._jax_enabled = False
            self._gpu_found = False

        self._hnswlib_enabled = False
        try:
            import hnswlib as _hnswlib

            self._hnswlib_enabled = True
            del _hnswlib
        except Exception:
            self._hnswlib_enabled = False

    def hnswlib_enabled(self):
        """
        Check if hnswlib is installed.

        Currently used to determine whether to call hnswlib API functions.

        Example:
            >>> from MuyGPyS import config
            >>> if config.hnswlib_enabled() is True:
            ...     import hnswlib
            ...     ...
            ... else:
            ...     ...
        """
        return self._hnswlib_enabled

    def gpu_found(self):
        """
        Check if jax is aware of gpus/tpus.

        Currently used to swap between numpy and jax implementations of core
        math functions in API files in cases where JAX is slower than numpy on
        CPU but faster on GPU. Notably, these include any math functions
        involving a linear solve.

        Example:
            >>> from MuyGPyS import config
            >>> if config.jax_enabled() is False:
            ...     from MuyGPyS._src.gp.numpy_muygps import _muygps_compute_solve
            ... else:
            ...     if config.gpu_found() is False:
            ...         from MuyGPyS._src.gp.numpy_muygps import _muygps_compute_solve
            ...     else:
            ...         from MuyGPyS._src.gp.jax_muygps import _muygps_compute_solve
        """
        return self._gpu_found

    def jax_enabled(self):
        """
        Check if jax is installed.

        Currently used to swap between numpy and jax implementations of core
        math functions in API files.

        Example:
            >>> from MuyGPyS import config
            >>> if config.jax_enabled() is False:
            ...     from MuyGPyS._src.gp.numpy_distance import _pairwise_distances
            ... else:
            ...     from MuyGPyS._src.gp.jax_distance import _pairwise_distances
        """
        return self._jax_enabled

    def x64_enabled(self):
        if jax_config is not None:
            return jax_config.x64_enabled
        else:
            return False

    def jax_enable_x64(self):
        """
        Elevate jax's internal registers to 64-bit words.

        Currently needed because x32 Matern matrices are not covariances!
        """
        if jax_config is not None:
            jax_config.update("jax_enable_x64", True)

    def jax_disable_x64(self):
        """
        Elevate jax's internal registers to 64-bit words.

        Currently needed because x32 Matern matrices are not covariances!
        """
        if jax_config is not None:
            jax_config.update("jax_enable_x64", False)

    def disable_jax(self):
        """
        Turn off jax support.

        Currently, must be run ahead of any imports that depend on
        Config.jax_enabled(). Currently only used for testing, but will make API
        more accesible if other uses are found.

        Example:
            >>> from MuyGPyS import config
            >>> config.disable_jax()
            >>> from MuyGPyS....  # other imports here
        """
        self._jax_enabled = False


config = Config()
config.jax_enable_x64()
