# Copyright 2021 Lawrence Livermore National Security, LLC and other MuyGPyS
# Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: (Apache-2.0 OR MIT)

import numpy as np

from scipy.special import gamma, kv


# class Matern:
#     """
#     The Matern kernel.

#     Parameters
#     ----------
#     X : numpy.ndarray(float), shape = ``(X_count, dim)''
#         A data matrix of locations.
#     X_star : numpy.ndarray(float), shape = ``(X_star_count, dim)''
#         A data matrix of alternative locations. If unspecified, compute
#         kernel on `X`.

#     Returns
#     -------
#     numpy.ndarray(float), shape = ``(X_count, X_star_count)''
#         The kernel matrix.
#     """

#     def __init__(self, nu=0.5, length_scale=10):
#         self.nu = nu
#         self.length_scale = length_scale

#     def __call__(self, D):
#         """
#         Compute kernel.

#         Parameters
#         ----------
#         X : numpy.ndarray(float), shape = ``(n, m)''
#             A distance matrix.

#         Returns
#         -------
#         numpy.ndarray(float), shape = ``(n, m)''
#             The kernel matrix.
#         """
#         pass


def cov2cor(A):
    """
    covariance matrix to correlation matrix.

    https://github.com/CamDavidsonPilon/Python-Numerics/blob/master/utils/cov2corr.py
    """
    d = np.sqrt(A.diagonal())
    A = ((A.T / d).T) / d
    np.fill_diagonal(A, 1.0)
    return A


def cov2varmean(cov):
    """
    Create a matrix of the geometric means of the variances in a covariance matrix

    cov: square array
        An NxN covariance matrix

    outputs
    -------
    varmean: square array
        sqrt{cov(x,x) * cov(x',x')}
    """
    d = np.sqrt(cov.diagonal(), dtype=np.float64)
    return np.outer(d, d)


class NNGP:
    def __init__(self, sigma_w_sq=0.5, sigma_b_sq=0.5, L=5):
        self.sigma_w_sq = sigma_w_sq
        self.sigma_b_sq = sigma_b_sq
        self.L = L

    def __call__(self, X, X_star=None):
        """
        Compute kernel.

        Parameters
        ----------
        X : numpy.ndarray(float), shape = ``(X_count, dim)''
            A data matrix of locations.
        X_star : numpy.ndarray(float), shape = ``(X_star_count, dim)''
            A data matrix of alternative locations. If unspecified, compute
            kernel on `X`.

        Returns
        -------
        numpy.ndarray(float), shape = ``(X_count, X_star_count)''
            The kernel matrix.
        """
        if X_star is None:
            X_star = X
        K = self.sigma_w_sq * X @ X_star.T + self.sigma_b_sq
        for _ in range(self.L):
            Kcorr = cov2cor(K)
            Kcorr[Kcorr > 1.0] = 1.0
            theta = np.arccos(Kcorr)
            K = cov2varmean(K) * (
                np.sin(theta) + (np.pi - theta) * np.cos(theta)
            )
            K *= self.sigma_w_sq / (2 * np.pi)
            K += self.sigma_b_sq
        return K


class Hyperparameter:
    def __init__(self, val, bounds):
        self.val = val
        self.bounds = bounds

    def _set(self, val=None, bounds=None):
        if val is not None:
            self.val = val
        if bounds is not None:
            self.bounds = bounds

    # def _set_bounds(self, bounds):
    #     self.bounds = bounds

    # def _set_val(self, val):
    #     self.val = val

    def get_bounds(self):
        return self.bounds

    def __call__(self):
        return self.val


def _init_hyperparameter(val_def, bounds_def, **kwargs):
    return Hyperparameter(
        kwargs.get("val", val_def), kwargs.get("bounds", bounds_def)
    )


class RBF:
    def __init__(self, length_scale=dict()):
        self.length_scale = _init_hyperparameter(
            1.0, (1e-5, 1e2), **length_scale
        )

    def __call__(self, squared_dists):
        """
        Compute kernels from distance tensor.

        Takes inspiration from
        [scikit-learn](https://github.com/scikit-learn/scikit-learn/blob/95119c13a/sklearn/gaussian_process/kernels.py#L1529)

        Parameters
        ----------
        squared_dists: numpy.ndarray(float),
                       shape = ``(data_count, nn_count, nn_count)'' or
                               ``(data_count, nn_count)''
            A matrix or tensor of pairwise squared l2 distances. Tensor
            diagonals along last two dimensions are expected to be 0.

        Returns
        -------
        numpy.ndarray(float), shape = ``(data_count, nn_count, nn_count)'' or
                                      ``(data_count, nn_count)''
            A kernel matrix or tensor of kernel matrices along the last two
            dimensions.
        """
        return np.exp(-squared_dists / (2 * self.length_scale() ** 2))

    def set_params(self, length_scale=dict()):
        self.length_scale._set(**length_scale)


class Matern:
    def __init__(self, nu=dict(), length_scale=dict()):
        self.nu = _init_hyperparameter(1.0, (1e-5, 2e1), **nu)
        self.length_scale = _init_hyperparameter(
            1.0, (1e-5, 2e1), **length_scale
        )

    def __call__(self, dists):
        nu = self.nu()
        # length_scale = self.length_scale()
        dists = dists / self.length_scale()
        if nu == 0.5:
            K = np.exp(-dists)
        elif nu == 1.5:
            K = dists * np.sqrt(3)
            K = (1.0 + K) * np.exp(-K)
        elif nu == 2.5:
            K = dists * np.sqrt(5)
            K = (1.0 + K + K ** 2 / 3.0) * np.exp(-K)
        elif nu == np.inf:
            K = np.exp(-(dists ** 2) / 2.0)
        else:
            K = dists
            K[K == 0.0] += np.finfo(float).eps
            tmp = np.sqrt(2 * nu) * K
            K.fill((2 ** (1.0 - nu)) / gamma(nu))
            K *= tmp ** nu
            K *= kv(nu, tmp)
        return K

    def set_params(self, nu=dict(), length_scale=dict()):
        self.nu._set(**nu)
        self.length_scale(**length_scale)
