# Copyright 2021 Lawrence Livermore National Security, LLC and other MuyGPyS
# Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: (Apache-2.0 OR MIT)

import numpy as np

from sklearn.gaussian_process.kernels import Matern, RBF

from MuyGPyS.gp.kernels import NNGPimpl as NNGP


class BenchmarkGP:
    """
    A basic Gaussian Process.

    Performs GP inference and simulation by way of analytic computations.
    """

    def __init__(self, kern="matern", **kwargs):
        """
        Initialize.

        Parameters
        ----------
        kern : str
            The kernel to be used. Each kernel supports different
            hyperparameters that can be specified in kwargs.
            NOTE[bwp] Currently supports ``matern'', ``rbf'' and ``nngp''.
        """
        self.kern = kern.lower()
        self.set_params(**kwargs)

    def set_params(self, **params):
        """
        Set the hyperparameters specified by `params`.

        NOTE[bwp] this logic should get moved into kernel functors once
        implemented

        Universal Parameters
        ----------
        eps : float
            The homoscedastic noise nugget to be added to the inverted
            covariance matrix.
        sigma_sq : np.ndarray(float), shape = ``(response_count)''
            Scaling parameter to be applied to posterior variance. One element
            per dimension of the response.

        Matern Parameters
        ----------
        nu : float
            The smoothness parameter. As ``nu'' -> infty, the matern kernel
            converges pointwise to the RBF kernel.
        length_scale : float
            Scale parameter multiplied against distance values.

        RBF Parameters
        ----------
        length_scale : float
            Scale parameter multiplied against distance values.

        NNGP Parameters
        ----------
        sigma_b_sq : float
            Variance prior on the bias parameters in a wide neural network under
            Glorot inigialization in the infinite width limit.
        sigma_w_sq : float
            Variance prior on the weight parameters in a wide neural network
            under Glorot inigialization in the infinite width limit.

        Returns
        -------
        unset_params : list(str)
            The set of kernel parameters that have not been fixed by ``params''.
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
        elif self.kern == "nngp":
            self.kernel = NNGP(
                sigma_b_sq=self.params.get("sigma_b_sq", 0.5),
                sigma_w_sq=self.params.get("sigma_w_sq", 0.5),
            )
            unset_params = {
                "eps",
                "sigma_sq",
                "sigma_b_sq",
                "sigma_w_sq",
            }.difference(params.keys())
        else:
            raise NotImplementedError(f"{self.kern} is not implemented yet!")
        return sorted(list(unset_params))

    def set_param_array(self, names, values):
        """
        Set the hyperparameters specified by elements of ``names'' with the
        corresponding elements of ``values''.

        Convenience function for use in concert with ``scipy.optimize''.

        NOTE[bwp] this logic should get moved into kernel functors once
        implemented

        Parameters
        ----------
        names : list(str)
            An alphabetically ordered list of parameter names.
        values : list(float)
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
        elif self.kern == "nngp":
            self.kernel = NNGP(**self.params)

    def optim_bounds(self, names, eps=1e-6):
        """
        Set the bounds (2-tuples) corresponding to each specified
        hyperparameter.

        NOTE[bwp] this logic should get moved into kernel functors once
        implemented

        Parameters
        ----------
        params : dict
            A dict mapping hyperparameter names to 2-tuples of floats. Floats
            must be increasing.
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
        elif self.kern == "nngp":
            if "sigma_b_sq" in names:
                ret.append((eps, 2.0))
            if "sigma_w_sq" in names:
                ret.append((eps, 2.0))
        return ret

    def fit(self, test, train):
        """
        Compute the full kernel and precompute the cholesky decomposition.

        Parameters
        ----------
        test : numpy.ndarray(float), shape = ``(test_count, feature_count)''
            The full testing data matrix.
        train : numpy.ndarray(float), shape = ``(train_count, feature_count)''
            The full training data matrix.
        """
        self._fit_kernel(np.vstack((test, train)))
        self.test_count = test.shape[0]
        self._cholesky(self.K)

    def fit_train(self, train):
        """
        Compute the training kernel and precompute the cholesky decomposition.

        Parameters
        ----------
        test : numpy.ndarray(float), shape = ``(test_count, dim)''
            The full testing data matrix.
        train : numpy.ndarray(float), shape = ``(train_count, dim)''
            The full training data matrix.
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
        targets,
        variance_mode=None,
    ):
        """
        Performs simultaneous regression on a list of observations.

        Parameters
        ----------
        variance_mode : str or None
            Specifies the type of variance to return. Currently supports
            ``diagonal'' and None. If None, report no variance term.
        targets : numpy.ndarray(float),
                  shape = ``(train_count, ouput_dim)''
            Vector-valued responses for each training element.

        Returns
        -------
        responses : numpy.ndarray(float),
                    shape = ``(batch_count, response_count,)''
            The predicted response for each of the given indices.
        diagonal_variance : numpy.ndarray(float), shape = ``(batch_count, )
            The diagonal elements of the posterior variance. Only returned where
            ``variance_mode == "diagonal"''.
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
