# Copyright 2021 Lawrence Livermore National Security, LLC and other MuyGPyS
# Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: (Apache-2.0 OR MIT)

import numpy as np

# from sklearn.gaussian_process.kernels import Matern, RBF

# from MuyGPyS.gp.kernels import NNGPimpl as NNGP
from MuyGPyS.gp.kernels import (
    Matern,
    RBF,
    NNGP,
    _get_kernel,
    _init_hyperparameter,
)


class MuyGPS:
    """
    Local Kriging Gaussian Process.

    Performs approximate GP inference by locally approximating an observation's
    response using its nearest neighbors.
    """

    def __init__(
        self, kern="matern", metric="l2", eps={}, sigma_sq={}, **kwargs
    ):
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
        self.metric = metric.lower()
        self.kernel = _get_kernel(self.kern, **kwargs)
        self.eps = _init_hyperparameter(1e-14, "fixed", **eps)
        self.sigma_sq = [
            _init_hyperparameter(1.0, "fixed", **ss) for ss in sigma_sq
        ]

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
        batch_size, nn_count = Kcross.shape
        _, response_count = targets.shape
        responses = Kcross.reshape(batch_size, 1, nn_count) @ np.linalg.solve(
            K + self.eps() * np.eye(nn_count), targets[nn_indices, :]
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
        _, response_count = targets.shape

        # K = self._compute_K(nn_indices, train)
        # sigmas = np.zeros((response_count,))
        for i in range(response_count):
            self.sigma_sq[i]._set_val(
                sum(self._get_sigma_sq(K, targets[:, i], nn_indices))
                / (nn_count * batch_size)
            )
        # self.sigma_sq = sigmas / (nn_count * batch_size)

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

        # K = self._compute_K(nn_indices, train)
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
                K[j, :, :] + self.eps() * np.eye(nn_count), Y_0
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
