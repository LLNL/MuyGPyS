# Copyright 2021 Lawrence Livermore National Security, LLC and other MuyGPyS
# Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: MIT

import numpy as np

from typing import Iterable, List, Optional, Tuple, Union

from sklearn.gaussian_process.kernels import Matern, RBF


class BenchmarkGP:
    """
    A basic Gaussian Process.

    Performs GP inference and simulation by way of analytic computations.

    Args:
        kern:
            The kernel to be used. Each kernel supports different
            hyperparameters that can be specified in kwargs.
            NOTE[bwp] Currently supports `matern` and `rbf`.
        **kwargs:
            Kernel parameters. See :ref:`MuyGPyS-gp-kernels`.
    """

    def __init__(
        self,
        kern: str = "matern",
        **kwargs,
    ):
        """
        Initialize.
        """
        self.kern = kern.lower()
        self.set_params(**kwargs)

    def set_params(self, **params) -> List[str]:
        """
        Set the hyperparameters specified by `params`.

        NOTE[bwp] this logic should get moved into kernel functors once
        implemented

        Args:
            eps:
                The homoscedastic noise nugget to be added to the inverted
                covariance matrix.
            sigma_sq:
                Scaling parameter to be applied to posterior variance. One
                element per dimension of the response.
            nu:
                The smoothness parameter. As `nu` -> infty, the matern kernel
                converges pointwise to the RBF kernel. Accept only if this is
                a matern GP.
            length_scale:
                Scale parameter multiplied against distance values. Accept if
                RBF or Matern.

        Returns:
            The set of kernel parameters that have not been fixed by `params`.
        """
        self.params = {
            p: params[p] for p in params if p != "eps" and p != "sigma_sq"
        }
        self.eps = params.get("eps", 0.015)
        self.sigma_sq = params.get("sigma_sq", np.array(1.0))
        if self.kern == "matern":
            self.kernel = Matern(
                length_scale=self.params.get("length_scale", 10.0),
                nu=self.params.get("nu", 0.5),
            )
            unset_params = {"eps", "sigma_sq", "length_scale", "nu"}.difference(
                params.keys()
            )
        elif self.kern == "rbf":
            self.kernel = RBF(length_scale=self.params.get("length_scale", 0.5))
            unset_params = {"eps", "sigma_sq", "length_scale"}.difference(
                params.keys()
            )
        else:
            raise NotImplementedError(f"{self.kern} is not implemented yet!")
        return sorted(list(unset_params))

    def set_param_array(
        self,
        names: List[str],
        values: List[float],
    ) -> None:
        """
        Set the hyperparameters specified by elements of `names` with the
        corresponding elements of `values`.

        Convenience function for use in concert with `scipy.optimize`.

        NOTE[bwp] this logic should get moved into kernel functors once
        implemented

        Args:
            names:
                An alphabetically ordered list of parameter names.
            values:
                A corresponding list of parameter values.
        """
        names = list(names)
        # this is going to break if we add a hyperparameter that occurs earlier
        # in alphabetical order.
        if names[0] == "eps":
            self.eps = values[0]
            names = names[1:]
            values = values[1:]
        for i, name in enumerate(names):
            self.params[name] = values[i]
        if self.kern == "matern":
            self.kernel = Matern(**self.params)
        elif self.kern == "rbf":
            self.kernel = RBF(**self.params)

    def optim_bounds(
        self,
        names: Iterable[str],
        eps: float = 1e-6,
    ) -> List[Tuple[float, float]]:
        """
        Set the bounds (2-tuples) corresponding to each specified
        hyperparameter.

        NOTE[bwp] this logic should get moved into kernel functors once
        implemented

        Args:
            names:
                An iterable over hyperparameter names.
        """
        ret = list()
        if "eps" in names:
            ret.append((eps, 0.2))
        if self.kern == "matern":
            if "length_scale" in names:
                ret.append((eps, 40.0))
            if "nu" in names:
                ret.append((eps, 2.0))
        elif self.kern == "rbf":
            if "length_scale" in names:
                ret.append((eps, 40.0))
        return ret

    def fit(
        self,
        test: np.ndarray,
        train: np.ndarray,
    ) -> None:
        """
        Compute the full kernel and precompute the cholesky decomposition.

        Args:
            test:
                The full testing data matrix of shape
                `(test_count, feature_count)`.
            train:
                The full training data matrix of shape
                `(train_count, feature_count)`.
        """
        self._fit_kernel(np.vstack((test, train)))
        self.test_count = test.shape[0]
        self._cholesky(self.K)

    def fit_train(self, train: np.ndarray) -> None:
        """
        Compute the training kernel and precompute the cholesky decomposition.

        Args:
            train:
                The full training data matrix of shape
                `(train_count, feature_count)`.
        """
        self._fit_kernel(train)
        self.test_count = 0
        self._cholesky(self.K)

    def _fit_kernel(self, x):
        self.K = self.kernel(x) + self.eps * np.eye(x.shape[0])

    def _cholesky(self, K):
        self.cholK = np.linalg.cholesky(K)

    def simulate(self):
        return self.cholK @ np.random.normal(0, 1, size=(self.cholK.shape[0],))

    def get_sigma_sq(self, y):
        assert y.shape[0] == self.K.shape[0]
        return (1 / y.shape[0]) * y @ np.linalg.solve(self.K, y)

    def regress(
        self,
        targets: np.ndarray,
        variance_mode: Optional[str] = None,
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        Performs simultaneous regression on a list of observations.

        Args:
            targets:
                A matrix of shape `(train_count, ouput_dim)` whose rows consist
                of vector-valued responses for each training element.
            variance_mode:
                Specifies the type of variance to return. Currently supports
                `diagonal` and None. If None, report no variance term.

        Returns
        -------
        responses:
            A matrix of shape `(batch_count, response_count)` whose rows consist
            of the predicted response for each of the given indices.
        diagonal_variance:
            A vector of shape `(batch_count,)` consisting of the diagonal
            elements of the posterior variance. Only returned where
            `variance_mode == "diagonal"`.
        """
        if self.test_count == 0:
            return np.array([])
        Kcross = self.K[self.test_count :, : self.test_count]
        K = self.K[: self.test_count, : self.test_count]
        responses = Kcross @ np.linalg.solve(K, targets)

        if variance_mode is None:
            return responses
        elif variance_mode == "diagonal":
            Kstar = self.K[self.test_count :, self.test_count :]
            variance = Kstar - Kcross @ np.linalg.solve(K, Kcross.T)
            return responses, np.diagonal(variance)
        elif variance_mode == "full":
            Kstar = self.K[self.test_count :, self.test_count :]
            variance = Kstar - Kcross @ np.linalg.solve(K, Kcross.T)
            return responses, variance
        else:
            raise NotImplementedError(
                f"Variance mode {variance_mode} is not implemented."
            )
