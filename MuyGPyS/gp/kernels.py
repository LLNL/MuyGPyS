# Copyright 2021 Lawrence Livermore National Security, LLC and other MuyGPyS
# Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: (Apache-2.0 OR MIT)

import numpy as np

from scipy.special import gamma, kv


def _get_kernel(kern, **kwargs):
    if kern == "rbf":
        return RBF(**kwargs)
    elif kern == "matern":
        return Matern(**kwargs)
    elif kern == "nngp":
        return NNGP(**kwargs)
    else:
        raise ValueError(f"Kernel type {self.kern} is not supported!")


class Hyperparameter:
    def __init__(self, val, bounds):
        self._set_bounds(bounds)
        self._set_val(val)

    def _set(self, val=None, bounds=None):
        if bounds is not None:
            self._set_bounds(bounds)
        if val is not None:
            self._set_val(val)

    def _set_val(self, val):
        if np.isscalar(val) is not True:
            raise ValueError(
                f"Nonscalar {val} of type {type(val)} is not a supported "
                f"hyperparameter type."
            )
        if self._bounds == "fixed":
            if val == "sample" or val == "log_sample":
                raise ValueError(
                    f"Must provide optimization bounds in order to sample a "
                    f"hyperparameter value."
                )
            if np.issubdtype(type(val), np.number) is not True:
                raise ValueError(
                    f"Unrecognized non-numeric type {type(val)} as "
                    f"hyperparamter value {val}."
                )
        else:
            if val == "sample":
                val = np.random.uniform(
                    low=self._bounds[0], high=self._bounds[1]
                )
            elif val == "log_sample":
                val = np.exp(
                    np.random.uniform(
                        low=np.log(self._bounds[0]),
                        high=np.log(self._bounds[1]),
                    )
                )
            else:
                if np.issubdtype(type(val), np.number) is not True:
                    raise ValueError(
                        f"Unrecognized non-numeric type {type(val)} as "
                        f"hyperparamter value {val}."
                    )
                if val < self._bounds[0]:
                    raise ValueError(
                        f"Hyperparameter value {val} is lesser than the "
                        f"optimization lower bound {self._bounds[0]}"
                    )
                if val > self._bounds[1]:
                    raise ValueError(
                        f"Hyperparameter value {val} is greater than the "
                        f"optimization upper bound {self._bounds[1]}"
                    )
        self._val = val

    def _set_bounds(self, bounds):
        if bounds != "fixed":
            if isinstance(bounds, str) is True:
                raise ValueError(f"Unknown bound option {bounds}.")
            if hasattr(bounds, "__iter__") is not True:
                raise ValueError(
                    f"Unknown bound optiom {bounds} of a non-iterable type "
                    f"{type(bounds)}."
                )
            if len(bounds) != 2:
                raise ValueError(
                    f"Provided hyperparameter optimization bounds have "
                    f"unsupported length {len(bounds)}."
                )
            if np.issubdtype(type(bounds[0]), np.number) is not True:
                raise ValueError(
                    f"Nonscalar {bounds[0]} of type {type(bounds[0])} is not a "
                    f"supported hyperparameter bound type."
                )
            if np.issubdtype(type(bounds[1]), np.number) is not True:
                raise ValueError(
                    f"Nonscalar {bounds[1]} of type {type(bounds[1])} is not a "
                    f"supported hyperparameter bound type."
                )
            if bounds[0] > bounds[1]:
                raise ValueError(
                    f"Lower bound {bounds[0]} is not lesser than upper bound "
                    f"{bounds[1]}."
                )
        self._bounds = bounds

    def __call__(self):
        return self._val

    def get_bounds(self):
        return self._bounds


def _init_hyperparameter(val_def, bounds_def, **kwargs):
    val = kwargs.get("val", val_def)
    bounds = kwargs.get("bounds", bounds_def)
    return Hyperparameter(val, bounds)


class KernelFn:
    def __init__(self, **kwargs):
        self.hyperparameters = dict()

    def set_params(self, **kwargs):
        for name in kwargs:
            self.hyperparameters[name]._set(**kwargs[name])

    def __str__(self):
        # Only for testing purposes.
        ret = ""
        for p in self.hyperparameters:
            param = self.hyperparameters[p]
            ret += f"{p} : {param()} - {param.get_bounds()}\n"
        return ret[:-1]


class RBF(KernelFn):
    def __init__(self, length_scale=dict(), metric="F2"):
        super().__init__()
        self.length_scale = _init_hyperparameter(1.0, "fixed", **length_scale)
        self.hyperparameters["length_scale"] = self.length_scale
        self.metric = metric

    def __call__(self, squared_dists):
        """
        Compute RBF kernels from distance tensor.

        Parameters
        ----------
        squared_dists: numpy.ndarray(float),
                       shape = ``(data_count, nn_count, nn_count)'' or
                               ``(data_count, nn_count)''
            A matrix or tensor of pairwise squared l2 distances. Matrix
            diagonals along last two dimensions are expected to be 0.

        Returns
        -------
        numpy.ndarray(float), shape = ``(data_count, nn_count, nn_count)'' or
                                      ``(data_count, nn_count)''
            A kernel matrix or tensor of kernel matrices along the last two
            dimensions.
        """
        return np.exp(-squared_dists / (2 * self.length_scale() ** 2))

    # def set_params(self, length_scale=dict()):
    #     self.length_scale._set(**length_scale)


class Matern(KernelFn):
    def __init__(self, nu=dict(), length_scale=dict(), metric="l2"):
        super().__init__()
        self.nu = _init_hyperparameter(1.0, "fixed", **nu)
        self.length_scale = _init_hyperparameter(1.0, "fixed", **length_scale)
        self.hyperparameters["nu"] = self.nu
        self.hyperparameters["length_scale"] = self.length_scale
        self.metric = metric

    def __call__(self, dists):
        """
        Compute Matern kernels from distance tensor.

        Takes inspiration from
        [scikit-learn](https://github.com/scikit-learn/scikit-learn/blob/95119c13a/sklearn/gaussian_process/kernels.py#L1529)

        Parameters
        ----------
        dists: numpy.ndarray(float),
                       shape = ``(data_count, nn_count, nn_count)'' or
                               ``(data_count, nn_count)''
            A matrix or tensor of pairwise l2 distances. Matrix diagonals along
            last two dimensions are expected to be 0.

        Returns
        -------
        numpy.ndarray(float), shape = ``(data_count, nn_count, nn_count)'' or
                                      ``(data_count, nn_count)''
            A kernel matrix or tensor of kernel matrices along the last two
            dimensions.
        """
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


class NNGP(KernelFn):
    def __init__(
        self, sigma_b_sq=dict(), sigma_w_sq=dict(), L=dict(), metric="ip"
    ):
        super().__init__()
        self.sigma_b_sq = _init_hyperparameter(0.5, "fixed", **sigma_b_sq)
        self.sigma_w_sq = _init_hyperparameter(0.5, "fixed", **sigma_w_sq)
        self.L = _init_hyperparameter(5, "fixed", **L)
        self.hyperparameters["sigma_b_sq"] = self.sigma_b_sq
        self.hyperparameters["sigma_w_sq"] = self.sigma_w_sq
        self.metric = metric

    def __call__(self, dists, nn_dists):
        """
        Compute NNGP kernels from distance tensors.

        Unlike RBF and Matern, implementation simultaneously requires both
        pairwise distances among (knn-induced) training data and between test
        and (knn-induced) training data.

        Parameters
        ----------
        dists: numpy.ndarray(float),
               shape = ``(data_count, nn_count, nn_count)''
            A tensor of pairwise cosine or inner product dissimilarities between
            all knn-induced training data. Tensor diagonals along last two
            dimensions are expected to be 0.
        nn_dists: numpy.ndarray(float), shape = ``(data_count, nn_count)''
            A matrix of pairwise cosine or inner product dissimilarities between
            testing data and knn-induced training data.

        Returns
        -------
        K : np.ndarray(float), shape = ``(data_count, nn_count, nn_count)''
            A tensor containing the ``nn_count'' x ``nn_count'' kernel matrices
            corresponding to each of the data elements.
        Kcross : np.ndarray(float), shape = ``(data_count, 1, nn_count)''
            A tensor containing the 1 x ``nn_count'' cross-covariance matrix
            corresponding to each of the data elements.
        """
        data_count, nn_count = nn_dists.shape
        full_dists = np.zeros((data_count, nn_count + 1, nn_count + 1))
        for i in range(data_count):
            full_dists[i, :-1, :-1] = dists[i, :, :]
            full_dists[i, -1, :-1] = nn_dists[i, :]
            full_dists[i, :-1, -1] = nn_dists[i, :]
            full_dists[i, -1, -1] = np.finfo(float).eps
        Kfull = np.array([self._kern(1.0 - mat) for mat in full_dists])
        K = Kfull[:, :-1, :-1]
        Kcross = Kfull[:, -1, :-1].reshape(data_count, 1, nn_count)
        return K, Kcross

    def _kern(self, X):
        # print(X)
        K = self.sigma_w_sq() * X + self.sigma_b_sq()
        # print(K)
        for _ in range(self.L()):
            Kcorr = cov2cor(K)
            # Kcorr[Kcorr > 1.0] = 1.0
            theta = np.arccos(Kcorr)
            K = cov2varmean(K) * (
                np.sin(theta) + (np.pi - theta) * np.cos(theta)
            )
            K *= self.sigma_w_sq() / (2 * np.pi)
            K += self.sigma_b_sq()
            # print(K)
        return K


class NNGPimpl:
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
        # print(X @ X_star.T)
        K = self.sigma_w_sq * X @ X_star.T + self.sigma_b_sq
        # print(K)
        for _ in range(self.L):
            Kcorr = cov2cor(K)
            Kcorr[Kcorr > 1.0] = 1.0
            theta = np.arccos(Kcorr)
            K = cov2varmean(K) * (
                np.sin(theta) + (np.pi - theta) * np.cos(theta)
            )
            K *= self.sigma_w_sq / (2 * np.pi)
            K += self.sigma_b_sq
            # print(K)
        return K


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
