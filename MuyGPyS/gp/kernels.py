# Copyright 2021 Lawrence Livermore National Security, LLC and other MuyGPyS 
# Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: (Apache-2.0 OR MIT)

import numpy as np


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
