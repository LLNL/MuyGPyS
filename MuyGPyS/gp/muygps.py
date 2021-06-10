# Copyright 2021 Lawrence Livermore National Security, LLC and other MuyGPyS
# Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: (Apache-2.0 OR MIT)

import numpy as np

from MuyGPyS.gp.distance import (
    crosswise_distances,
    pairwise_distances,
)
from MuyGPyS.gp.kernels import (
    _get_kernel,
    _init_hyperparameter,
)


class MuyGPS:
    """
    Local Kriging Gaussian Process.

    Performs approximate GP inference by locally approximating an observation's
    response using its nearest neighbors.

    Kernels accept different hyperparameter dictionaries specifying
    hyperparameter settings. Keys can include ``val'' and ``bounds''.
    ``bounds'' must be either a len == 2  iterable container whose elements
    are scalars in increasing order, or the string ``fixed''. If
    ``bounds == fixed'' (the default behavior), the hyperparameter value
    will remain fixed during optimization. ``val'' must be either a scalar
    (within the range of the upper and lower bounds if given) or the strings
    ``sample'' or ``log_sample'', which will randomly sample a value within
    the range given by the bounds.

    In addition to individual kernel hyperparamters, each MuyGPS object also
    possesses a homoscedastic :math:`\\varepsilon` noise parameter and a
    vector of :math:`\\sigma^2`.

    Parameters
    ----------
    kern : str
        The kernel to be used. Each kernel supports different
        hyperparameters that can be specified in kwargs.
        NOTE[bwp] Currently supports only ``matern'' and ``rbf''.
    eps : dict
        A hyperparameter dict.
    sigma_sq : Iterable(dicts)
        An iterable container of hyperparameter dicts.
    kwargs : dict
        Addition parameters to be passed to the kernel, possibly including
        additional hyperparameter dicts and a metric keyword.
    """

    def __init__(
        self,
        kern="matern",
        eps={"val": 1e-5},
        sigma_sq=[{"val": 1e0}],
        **kwargs,
    ):
        self.kern = kern.lower()
        self.kernel = _get_kernel(self.kern, **kwargs)
        self.eps = _init_hyperparameter(1e-14, "fixed", **eps)
        self.sigma_sq = [
            _init_hyperparameter(1.0, "fixed", **ss) for ss in sigma_sq
        ]

    def fixed(self):
        """
        Checks whether all kernel and model parameters are fixed.

        This is a convenience utility to determine whether optimization is
        required.

        Returns
        -------
        bool
            Returns ``True'' if all parameters are fixed, and false otherwise.
        """
        return self.fixed_nosigmasq() and self.fixed_sigmasq()

    def fixed_nosigmasq(self):
        """
        Checks whether all kernel and model parameters are fixed, excluding
        :math:`\\sigma^2`.

        Returns
        -------
        bool
            Returns ``True'' if all parameters are fixed, and false otherwise.
        """
        for p in self.kernel.hyperparameters:
            if self.kernel.hyperparameters[p].get_bounds() != "fixed":
                return False
        if self.eps.get_bounds() != "fixed":
            return False
        return True

    def fixed_sigmasq(self):
        """
        Checks whether all dimensions of :math:`\\sigma^2` are fixed.

        Returns
        -------
        bool
            Returns ``True'' if all :math:`\\sigma^2` dimensions are fixed, and
            false otherwise.
        """
        for ss in self.sigma_sq:
            if ss.get_bounds() != "fixed":
                return False
        return True

    def get_optim_params(self):
        """
        Return a dictionary of references to the unfixed kernel hyperparameters.

        This is a convenience function for obtaining all of the information
        necessary to optimize hyperparameters. It is important to note that the
        values of the dictionary are references to the actual hyperparameter
        objects underying the kernel functor - changing these references will
        change the kernel.

        Returns
        -------
        dict (str: MuyGPyS.gp.kernel.Hyperparameter)
            A dict mapping hyperparameter names to references to their objects.
            Only returns hyperparameters whose bounds are not set as ``fixed''.
            Returned hyperparameters can include ``eps'', but not ``sigma_sq'',
            as it is currently optimized via a separate closed-form method.
        """
        optim_params = {
            p: self.kernel.hyperparameters[p]
            for p in self.kernel.hyperparameters
            if self.kernel.hyperparameters[p].get_bounds() != "fixed"
        }
        if self.eps.get_bounds() != "fixed":
            optim_params["eps"] = self.eps
        return optim_params

    def _compute_solve(self, K, Kcross, batch_targets):
        """
        Simultaneously solve all of the GP inference systems of linear
        equations.

        Parameters
        ----------
        K : np.ndarray(float), shape = ``(batch_size, nn_count, nn_count)''
            A tensor containing the ``nn_count'' x ``nn_count'' kernel matrices
            corresponding to each of the batch elements.
        Kcross : np.ndarray(float), shape = ``(batch_size, nn_count)''
            A tensor containing the 1 x ``nn_count'' cross-covariance matrix
            corresponding to each of the batch elements.
        batch_targets : numpy.ndarray(float),
                  shape = ``(batch_size, nn_count, response_count)''
            The vector-valued responses for the nearest neighbors of each
            batch element.

        Returns
        -------
        numpy.ndarray(float), shape = ``(batch_count, response_count)''
            The predicted response for each of the given indices.
        """
        batch_size, nn_count, response_count = batch_targets.shape
        responses = Kcross.reshape(batch_size, 1, nn_count) @ np.linalg.solve(
            K + self.eps() * np.eye(nn_count), batch_targets
        )
        return responses.reshape(batch_size, response_count)

    def _compute_diagonal_variance(self, K, Kcross):
        """
        Simultaneously solve all of the GP inference systems of linear
        equations.

        Parameters
        ----------
        K : np.ndarray(float), shape = ``(batch_size, nn_count, nn_count)''
            A tensor containing the ``nn_count'' x ``nn_count'' kernel matrices
            corresponding to each of the batch elements.
        Kcross : np.ndarray(float), shape = ``(batch_size, nn_count)''
            A tensor containing the 1 x ``nn_count'' cross-covariance matrix
            corresponding to each of the batch elements.

        Returns
        -------
        numpy.ndarray(float), shape = ``(batch_count, response_count,)''
            The predicted response for each of the given indices.
        """
        batch_size, nn_count = Kcross.shape
        return np.array(
            [
                1.0
                - Kcross[i, :]
                @ np.linalg.solve(
                    K[i, :, :] + self.eps() * np.eye(nn_count), Kcross[i, :]
                )
                for i in range(batch_size)
            ]
        )

    def regress_from_indices(
        self,
        indices,
        nn_indices,
        test,
        train,
        targets,
        variance_mode=None,
    ):
        """
        Performs simultaneous regression on a list of observations.

        This is similar to the old regress API in that it implicitly creates and
        discards the distance and kernel matrices.

        Parameters
        ----------
        indices : np.ndarray(int), shape = ``(batch_count,)''
            The integer indices of the observations to be approximated.
        nn_indices : numpy.ndarray(int), shape = ``(batch_size, nn_count)''
            A matrix listing the nearest neighbor indices for all observations
            in the testing batch.
        train : numpy.ndarray(float), shape = ``(train_count, feature_count)''
            The full training data matrix.
        test : numpy.ndarray(float), shape = ``(test_count, feature_count)''
            The full testing data matrix.
        targets : numpy.ndarray(float),
                  shape = ``(train_count, response_count)''
            Vector-valued responses for each training element.
        variance_mode : str or None
            Specifies the type of variance to return. Currently supports
            ``diagonal'' and None. If None, report no variance term.

        Returns
        -------
        responses : numpy.ndarray(float),
                    shape = ``(batch_count, response_count,)''
            The predicted response for each of the given indices.
        diagonal_variance : numpy.ndarray(float), shape = ``(batch_count,)
            The diagonal elements of the posterior variance. Only returned where
            ``variance_mode == "diagonal"''.
        """
        crosswise_dists = crosswise_distances(
            test, train, indices, nn_indices, metric=self.kernel.metric
        )
        pairwise_dists = pairwise_distances(
            train, nn_indices, metric=self.kernel.metric
        )
        K = self.kernel(pairwise_dists)
        Kcross = self.kernel(crosswise_dists)
        batch_targets = targets[nn_indices, :]
        return self.regress(
            K, Kcross, batch_targets, variance_mode=variance_mode
        )

    def regress(
        self,
        K,
        Kcross,
        batch_targets,
        variance_mode=None,
    ):
        """
        Performs simultaneous regression on provided covariance,
        cross-covariance, and target.

        Parameters
        ----------
        K : np.ndarray(float), shape = ``(batch_size, nn_count, nn_count)''
            A tensor containing the ``nn_count'' x ``nn_count'' kernel matrices
            corresponding to each of the batch elements.
        Kcross : np.ndarray(float), shape = ``(batch_size, nn_count)''
            A tensor containing the 1 x ``nn_count'' cross-covariance matrix
            corresponding to each of the batch elements.
        batch_targets : numpy.ndarray(float),
                  shape = ``(batch_size, nn_count, response_count)''
            The vector-valued responses for the nearest neighbors of each
            batch element.
        variance_mode : str or None
            Specifies the type of variance to return. Currently supports
            ``diagonal'' and None. If None, report no variance term.

        Returns
        -------
        responses : numpy.ndarray(float),
                    shape = ``(batch_count, response_count,)''
            The predicted response for each of the given indices.
        diagonal_variance : numpy.ndarray(float), shape = ``(batch_count,)
            The diagonal elements of the posterior variance. Only returned where
            ``variance_mode == "diagonal"''.
        """
        responses = self._compute_solve(K, Kcross, batch_targets)
        if variance_mode is None:
            return responses
        elif variance_mode == "diagonal":
            diagonal_variance = self._compute_diagonal_variance(K, Kcross)
            return responses, diagonal_variance
        else:
            raise NotImplementedError(
                f"Variance mode {variance_mode} is not implemented."
            )

    def sigma_sq_optim(
        self,
        K,
        nn_indices,
        targets,
    ):
        """
        Optimize the value of the sigma^2 scale parameter for each response
        dimension.

        We approximate sigma^2 by way of averaging over the analytic solution
        from each local kernel.

        sigma^2 = 1/n * Y^T @ K^{-1} @ Y

        Parameters
        ----------
        index : np.ndarray(int), shape = ``(batch_count,)''
            The integer indices of the observations to be approximated.
        nn_indices : numpy.ndarray(int), shape = ``(batch_size, nn_count)''
            A matrix listing the nearest neighbor indices for all observations
            in the testing batch.
        targets : numpy.ndarray(float),
                  shape = ``(train_count, response_count)''
            Vector-valued responses for each training element.

        Returns
        -------
        sigmas : numpy.ndarray(float), shape = ``(response_count,)''
            The value of sigma^2 for each dimension.
        """
        batch_size, nn_count = nn_indices.shape
        _, response_count = targets.shape

        for i in range(response_count):
            self.sigma_sq[i]._set_val(
                sum(self._get_sigma_sq(K, targets[:, i], nn_indices))
                / (nn_count * batch_size)
            )

    def _get_sigma_sq_series(
        self,
        K,
        nn_indices,
        target_col,
    ):
        """
        Return the series of sigma^2 scale parameters for each neighborhood
        solve.
        NOTE[bwp]: This function is only for testing purposes.

        Parameters
        ----------
        index : np.ndarray(int), shape = ``(batch_count,)''
            The integer indices of the observations to be approximated.
        nn_indices : numpy.ndarray(int), shape = ``(batch_size, nn_count)''
            A matrix listing the nearest neighbor indices for all observations
            in the testing batch.
        train : numpy.ndarray(float), shape = ``(train_count, feature_count)''
            The full training data matrix.
        target_col : numpy.ndarray(float), shape = ``(train_count,)''
            The target vector consisting of the target for each nearest
            neighbor.

        Returns
        -------
        sigmas : numpy.ndarray(float), shape = ``(response_count,)''
            The value of sigma^2 for each dimension.
        """
        batch_size, nn_count = nn_indices.shape

        sigmas = np.zeros((batch_size,))
        for i, el in enumerate(self._get_sigma_sq(K, target_col, nn_indices)):
            sigmas[i] = el
        return sigmas / nn_count

    def _get_sigma_sq(self, K, target_col, nn_indices):
        """
        Generate series of :math:`\\sigma^2` scale parameters for each
        individual solve along a single dimension:

        .. math::
            \\sigma^2 = \\frac{1}{k} * Y_{nn}^T K_{nn}^{-1} Y_{nn}

        Here :math:`Y_{nn}` and :math:`K_{nn}` are the target and kernel
        matrices with respect to the nearest neighbor set in scope, where
        :math:`k` is the number of nearest neighbors.

        Parameters
        ----------
        K : np.ndarray(float), shape = ``(batch_count, nn_count, nn_count)''
            Kernel tensor containing nearest neighbor kernels for each local
            neighborhood.
        target_col : numpy.ndarray(float), shape = ``(batch_count,)''
            The target vector consisting of the target for each nearest
            neighbor.
        nn_indices : numpy.ndarray(int), shape = ``(batch_size, nn_count)''
            A matrix listing the nearest neighbor indices for all observations
            in the testing batch.

        Yields
        -------
        sigmas : numpy.ndarray(float), shape = ``(batch_count,)''
            The optimal value of sigma^2 for each neighborhood for the given
            output dimension.
        """
        batch_size, nn_count = nn_indices.shape
        for j in range(batch_size):
            Y_0 = target_col[nn_indices[j, :]]
            yield Y_0 @ np.linalg.solve(
                K[j, :, :] + self.eps() * np.eye(nn_count), Y_0
            )
