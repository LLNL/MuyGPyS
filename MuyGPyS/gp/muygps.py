#!/usr/bin/env python
# encoding: utf-8
"""
@file muygps.py

Created by priest2 on 2020-10-19

Implements the local kriging MuyGPS approximation logic.
"""

import numpy as np

from sklearn.gaussian_process.kernels import Matern, RBF

from MuyGPyS.gp.kernels import NNGP


class MuyGPS:
    """
    Local Kriging Gaussian Process.

    Performs approximate GP inference by locally approximating an observation's
    response using its nearest neighbors.
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
        self.bounds = dict()

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

    def set_optim_bounds(self, **params):
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
        for p in params:
            assert len(params[p]) == 2
            self.bounds[p] = params[p]

    def _get_bound(self, param, default):
        """
        Return the optimization bounds corresponding to the given
        hyperparameter.

        NOTE[bwp] this logic should get moved into kernel functors once
        implemented

        Parameters
        ----------
        param : str
            A hyperparameter name.
        default : tuple(float), shape = (2,)
            Return value if ``param'' is not set in ``self.bounds''.

        Returns
        -------
        tuple(float), shape = (2,)
            A pair of (min, max) values to be used for hyperparameter
            optimization.
        """
        return self.bounds.get(param, default)

    def optim_bounds(self, names, eps=1e-6):
        """
        Return hyperparameter bounds.

        NOTE[bwp] this logic should get moved into kernel functors once
        implemented

        Parameters
        ----------
        names : list(str)
            The set of hyperparameter names to be queried.

        Returns
        -------
        names : list(str)
            The set of hyperparameter names to be queried.
        """
        ret = list()
        if "eps" in names:
            ret.append(self.bounds.get("eps", (eps, 0.2)))
        if self.kern == "matern":
            if "length_scale" in names:
                ret.append(self.bounds.get("length_scale", (eps, 40.0)))
            if "nu" in names:
                ret.append(self.bounds.get("nu", (eps, 2.0)))
        elif self.kern == "rbf":
            if "length_scale" in names:
                ret.append(self.bounds.get("length_scale", (eps, 40.0)))
        elif self.kern == "nngp":
            if "sigma_b_sq" in names:
                ret.append(self.bounds.get("sigma_b_sq", (eps, 2.0)))
            if "sigma_w_sq" in names:
                ret.append(self.bounds.get("sigma_w_sq", (eps, 2.0)))
        return ret

    def _compute_K(self, nn_indices, train):
        """
        Compute the Kernel tensor.

        NOTE[bwp] this will be reimplemented once kernels/distances are
        extracted.

        Parameters
        ----------
        nn_indices : numpy.ndarray(int), shape = ``(batch_size, nn_count)''
            A matrix listing the nearest neighbor indices for all observations
            in the testing batch.
        train : numpy.ndarray(float), shape = ``(train_count, feature_count)''
            The full training data matrix.

        Returns
        -------
        np.ndarray(float), shape = ``(batch_size, nn_count, nn_count)''
            A tensor containing the ``nn_count'' x ``nn_count'' kernel matrices
            corresponding to each of the batch elements.
        """
        return np.array([self.kernel(mat) for mat in train[nn_indices]])

    def _compute_Kcross(self, indices, nn_indices, test, train):
        """
        Compute the cross-covariance tensor.

        NOTE[bwp] this will be reimplemented once kernels/distances are
        extracted.

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

        Returns
        -------
        np.ndarray(float), shape = ``(batch_size, 1, nn_count)''
            A tensor containing the 1 x ``nn_count'' cross-covariance matrix
            corresponding to each of the batch elements.
        """
        feature_count = test.shape[1]
        return np.array(
            [
                self.kernel(vec.reshape(1, feature_count), mat)
                for vec, mat in zip(test[indices], train[nn_indices])
            ]
        )

    def _compute_Kfull(self, indices, nn_indices, test, train):
        """
        Compute the full NNGP Kernel tensor.

        NOTE[bwp] this will be reimplemented once kernels/distances are
        extracted.

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

        Returns
        -------
        np.ndarray(float), shape = ``(batch_size, nn_count + 1, nn_count + 1)''
            A tensor containing the ``nn_count + 1'' x ``nn_count + 1''
            full covariance matrix corresponding to each of the batch elements.
        """
        feature_count = test.shape[1]
        return np.array(
            [
                self.kernel(np.vstack((mat, vec.reshape(1, feature_count))))
                for vec, mat in zip(test[indices], train[nn_indices])
            ]
        )

    def _compute_kernel_tensors(
        self,
        indices,
        nn_indices,
        test,
        train,
    ):
        """
        Compute the kernel and cross-covariance tensors for all batch elements.

        NOTE[bwp] implementation is split due to NNGP implementation.

        NOTE[bwp] this will be reimplemented once kernels/distances are
        extracted.

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

        Returns
        -------
        K : np.ndarray(float), shape = ``(batch_size, nn_count, nn_count)''
            A tensor containing the ``nn_count'' x ``nn_count'' kernel matrices
            corresponding to each of the batch elements.
        Kcross : np.ndarray(float), shape = ``(batch_size, 1, nn_count)''
            A tensor containing the 1 x ``nn_count'' cross-covariance matrix
            corresponding to each of the batch elements.
        """
        # NOTE[bwp] This is clugy and terrible. Need to reenginer nngp.
        # NOTE[bwp] In fact, should reengineer all kernels so as to use on-node
        # parallelism. This is one of the main bottlenecks right now.

        if type(self.kernel) == NNGP:
            batch_size, nn_count = nn_indices.shape
            Kfull = self._compute_Kfull(indices, nn_indices, test, train)
            K = Kfull[:, :-1, :-1]
            Kcross = Kfull[:, -1, :-1].reshape((batch_size, 1, nn_count))
        else:
            K = self._compute_K(nn_indices, train)
            Kcross = self._compute_Kcross(indices, nn_indices, test, train)
        return K, Kcross

    def _compute_solve(self, nn_indices, targets, K, Kcross):
        """
        Simultaneously solve all of the GP inference systems of linear
        equations.

        Parameters
        ----------
        nn_indices : numpy.ndarray(int), shape = ``(batch_size, nn_count)''
            A matrix listing the nearest neighbor indices for all observations
            in the testing batch.
        targets : numpy.ndarray(float),
                  shape = ``(train_count, response_count)''
            Vector-valued responses for each training element.
        K : np.ndarray(float), shape = ``(batch_size, nn_count, nn_count)''
            A tensor containing the ``nn_count'' x ``nn_count'' kernel matrices
            corresponding to each of the batch elements.
        Kcross : np.ndarray(float), shape = ``(batch_size, 1, nn_count)''
            A tensor containing the 1 x ``nn_count'' cross-covariance matrix
            corresponding to each of the batch elements.

        Returns
        -------
        numpy.ndarray(float), shape = ``(batch_count, response_count)''
            The predicted response for each of the given indices.
        """
        batch_size, nn_count = nn_indices.shape
        response_count = targets.shape[1]
        responses = Kcross @ np.linalg.solve(
            K + self.eps * np.eye(nn_count), targets[nn_indices, :]
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
        Kcross : np.ndarray(float), shape = ``(batch_size, 1, nn_count)''
            A tensor containing the 1 x ``nn_count'' cross-covariance matrix
            corresponding to each of the batch elements.

        Returns
        -------
        numpy.ndarray(float), shape = ``(batch_count, response_count,)''
            The predicted response for each of the given indices.
        """
        batch_size, _, nn_count = Kcross.shape
        return np.array(
            [
                1.0
                - Kcross[i, 0, :]
                @ np.linalg.solve(
                    K[i, :, :] + self.eps * np.eye(nn_count), Kcross[i, 0, :]
                )
                for i in range(batch_size)
            ]
        )

    def regress(
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
        nn_count = nn_indices.shape[1]
        K, Kcross = self._compute_kernel_tensors(
            indices,
            nn_indices,
            test,
            train,
        )
        responses = self._compute_solve(nn_indices, targets, K, Kcross)
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
        indices,
        nn_indices,
        train,
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
        train : numpy.ndarray(float), shape = ``(train_count, feature_count)''
            The full training data matrix.
        targets : numpy.ndarray(float),
                  shape = ``(train_count, response_count)''
            Vector-valued responses for each training element.

        Returns
        -------
        sigmas : numpy.ndarray(float), shape = ``(response_count,)''
            The value of sigma^2 for each dimension.
        """
        batch_size, nn_count = nn_indices.shape
        response_count = targets.shape[1]

        K = self._compute_K(nn_indices, train)
        sigmas = np.zeros((response_count,))
        for i in range(response_count):
            sigmas[i] = sum(self._get_sigma_sq(K, targets[:, i], nn_indices))
        self.sigma_sq = sigmas / (nn_count * batch_size)

    def get_sigma_sq(
        self,
        indices,
        nn_indices,
        train,
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

        K = self._compute_K(nn_indices, train)
        sigmas = np.zeros((batch_size,))
        for i, el in enumerate(self._get_sigma_sq(K, target_col, nn_indices)):
            sigmas[i] = el
        return sigmas / nn_count

    def _get_sigma_sq(self, K, target_col, nn_indices):
        """
        Generate series of sigma^2 scale parameters for each individual solve
        along a single dimension:

        sigma^2 = 1/nn * Y_{nn}^T @ K_{nn}^{-1} @ Y_{nn}

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
                K[j, :, :] + self.eps * np.eye(nn_count), Y_0
            )


#     # def do_GP(
#     #     kernel,
#     #     test_indices,
#     #     nn_indices,
#     #     embedded_test,
#     #     embedded_train,
#     #     train_diags,
#     #     test_diags,
#     # ):
#     # dist = 1 - (
#     #     (embedded_train[coords2, :] @ embedded_train[coords2, :].T)
#     #     / np.outer(
#     #         np.sqrt(train_diags[coords2]), np.sqrt(train_diags[coords2])
#     #     )
#     # )
#     # dist[dist < 0] = 0.0
#     # cross_dist = 1 - (
#     #     (embedded_test[test_index, :] @ embedded_train[coords2, :].T)
#     #     / np.sqrt(train_diags[coords2])
#     #     * np.sqrt(test_diags[test_index])
#     # )
#     # cross_dist[cross_dist < 0] = 0
#     # K = kernel(dist)
#     # Kcross = kernel(cross_dist)

#     # label = np.argmax(
#     #     Kcross
#     #     @ np.linalg.solve(K + eps * np.eye(K.shape[0]), labels[coords2, :])
#     # )
#     # return label


# def do_GP_tensor(
#     nu,
#     test_indices,
#     nn_indices,
#     embedded_test,
#     embedded_train,
#     train_labels,
#     eps=0.015,
#     kern="matern",
# ):
#     """
#     Using https://stackoverflow.com/questions/26089893/understanding-numpys-einsum
#     """
#     #     diff_tensor = 1 - np.einsum('bij, bjk -> bik',
#     #                                 embedded_train[batch_nn_indices],
#     #                                 embedded_train[batch_nn_indices].transpose(0,2,1))
#     #     diag_tensor = np.einsum("bi, bj -> bij",
#     #                             np.sqrt(train_diags[batch_nn_indices]),
#     #                             np.sqrt(train_diags[batch_nn_indices]))
#     #     dist_tensor = diff_tensor / diag_tensor
#     #     dist_tensor[dist_tensor < 0] = 0.

#     #     cross_diff_tensor = 1 - np.einsum('bj, bij -> bi',
#     #                                       embedded_test[test_indices],
#     #                                       embedded_train[batch_nn_indices])
#     #     cross_diag_tensor = np.einsum("b, bi -> bi",
#     #                                   np.sqrt(test_diags[test_indices]),
#     #                                   np.sqrt(train_diags[batch_nn_indices]))
#     #     cross_dist_tensor = cross_diff_tensor / cross_diag_tensor
#     #     cross_dist_tensor[cross_dist_tensor < 0] = 0.
