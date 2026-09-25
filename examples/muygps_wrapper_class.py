import numpy as np
from time import perf_counter
from typing import Dict, Tuple
from MuyGPyS.examples.from_indices import regress_from_indices, tensors_from_indices
from MuyGPyS.gp import MuyGPS 
from MuyGPyS.neighbors import NN_Wrapper
from MuyGPyS.optimize import Bayes_optimize, OptimizeFn
from MuyGPyS.optimize.batch import sample_batch
from MuyGPyS.optimize.loss import LossFn, lool_fn
from MuyGPyS.optimize.loss import *

class MuyGPSWrapper(MuyGPS):

    def fit_regressor(
        self,
        train_features: np.ndarray,
        train_targets: np.ndarray,
        nn_count: int = 30,
        batch_count: int = 200,
        loss_fn: LossFn = lool_fn,
        opt_fn: OptimizeFn = Bayes_optimize,
        nn_kwargs: Dict = dict(),
        opt_kwargs: Dict = dict(),
        verbose: bool = False,
        ) -> Tuple[MuyGPS, NN_Wrapper]:
        """
        Convenience function for creating MuyGPyS functor and neighbor lookup data
        structure.

        Expected parameters include keyword argument dicts specifying kernel
        parameters and nearest neighbor parameters. See the docstrings of the
        appropriate functions for specifics.

        Example:
            >>> from MuyGPyS.examples.regress import make_regressor
            >>> from MuyGPyS.gp.deformation import F2, Isotropy
            >>> from MuyGPyS.gp.hyperparameter import Parameter
            >>> from MuyGPyS.gp.hyperparameter import AnalyticScale
            >>> from MuyGPyS.gp.kernels import RBF
            >>> from MuyGPyS.gp.noise import HomoscedasticNoise
            >>> from MuyGPyS.optimize import Bayes_optimize
            >>> from MuyGPyS.examples.regress import make_regressor
            >>> train_features, train_responses = make_train()  # stand-in function
            >>> nn_kwargs = {"nn_method": "exact", "algorithm": "ball_tree"}
            >>> muygps, nbrs_lookup = make_regressor(
            ...         train_features,
            ...         train_responses,
            ...         nn_count=30,
            ...         batch_count=200,
            ...         loss_fn=lool_fn,
            ...         opt_fn=Bayes_optimize,
            ...         k_kwargs=k_kwargs,
            ...         nn_kwargs=nn_kwargs,
            ...         verbose=False,
            ... )

        Args:
            train_features:
                A matrix of shape `(train_count, feature_count)` whose rows consist
                of observation vectors of the train data.
            train_targets:
                A matrix of shape `(train_count, response_count)` whose rows consist
                of response vectors of the train data.
            nn_count:
                The number of nearest neighbors to employ.
            batch_count:
                The number of elements to sample batch for hyperparameter
                optimization.
            loss_fn:
                The loss method to use in hyperparameter optimization. Ignored if
                all of the parameters specified by argument `k_kwargs` are fixed.
            opt_fn:
                The optimization functor to use in hyperparameter optimization.
                Ignored if all of the parameters specified by argument `k_kwargs`
                are fixed.
            nn_kwargs:
                Parameters for the nearest neighbors wrapper. See
                :class:`MuyGPyS.neighbors.NN_Wrapper` for the supported methods and
                their parameters.
            opt_kwargs:
                Parameters for the wrapped optimizer. See the docs of the
                corresponding library for supported parameters.
            verbose:
                If `True`, print summary statistics.

        Returns
        -------
        muygps:
            A (possibly trained) MuyGPs object.
        nbrs_lookup:
            A data structure supporting nearest neighbor queries into
            `train_features`.
        """
        train_count = train_features.shape[0]
        time_start = perf_counter()

        nbrs_lookup = NN_Wrapper(
            train_features,
            nn_count,
            **nn_kwargs,
        )
        time_nn = perf_counter()

        skip_opt = self.fixed()
        if skip_opt is False:
            # collect batch
            batch_indices, batch_nn_indices = sample_batch(
                nbrs_lookup,
                batch_count,
                train_count,
            )
            time_batch = perf_counter()

            (
                crosswise_diffs,
                pairwise_diffs,
                batch_targets,
                batch_nn_targets,
            ) = self.make_train_tensors(
                batch_indices,
                batch_nn_indices,
                train_features,
                train_targets,
            )
            time_tensor = perf_counter()

            if skip_opt is False:
                # maybe do something with these estimates?
                self = opt_fn(
                    self,
                    batch_targets,
                    batch_nn_targets,
                    crosswise_diffs,
                    pairwise_diffs,
                    loss_fn=loss_fn,
                    verbose=verbose,
                    **opt_kwargs,
                )
            time_opt = perf_counter()

            self = self.optimize_scale(pairwise_diffs, batch_nn_targets)
            if verbose is True:
                print(f"Optimized scale values " f"{self.scale()}")
            time_sopt = perf_counter()

            if verbose is True:
                print(f"NN lookup creation time: {time_nn - time_start}s")
                print(f"batch sampling time: {time_batch - time_nn}s")
                print(f"tensor creation time: {time_tensor - time_batch}s")
                print(f"hyper opt time: {time_opt - time_tensor}s")
                print(f"scale opt time: {time_sopt - time_opt}s")

        return self, nbrs_lookup

    def regress_any(
        self,
        test_features: np.ndarray,
        train_features: np.ndarray,
        train_nbrs_lookup: NN_Wrapper,
        train_targets: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Simultaneously predicts the response for each test item.

        Args:
            self:
                Regressor object.
            test_features:
                Test observations of shape `(test_count, feature_count)`.
            train_features:
                Train observations of shape `(train_count, feature_count)`.
            train_nbrs_lookup:
                Trained nearest neighbor query data structure.
            train_targets:
                Observed response for all training data of shape
                `(train_count, class_count)`.

        Returns
        -------
        means:
            The predicted response of shape `(test_count, response_count,)` for
            each of the test examples.
        variances:
            The independent posterior variances for each of the test examples of
            shape `(test_count,)`.
        """
        test_count = test_features.shape[0]
        test_nn_indices, _ = train_nbrs_lookup.get_nns(test_features)

        posterior_mean, posterior_variance = regress_from_indices(
            self,
            np.arange(test_count),
            test_nn_indices,
            test_features,
            train_features,
            train_targets,
        )

        return posterior_mean, posterior_variance
    
    def regress_one(
        self,
        test_features: np.ndarray,
        train_features: np.ndarray,
        train_nbrs_lookup: NN_Wrapper,
        train_targets: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Simultaneously predicts the response for each test item.

        Args:
            self:
                Regressor object.
            test_features:
                Test observations of shape `(test_count, feature_count)`.
            train_features:
                Train observations of shape `(train_count, feature_count)`.
            train_nbrs_lookup:
                Trained nearest neighbor query data structure.
            train_targets:
                Observed response for all training data of shape
                `(train_count, class_count)`.

        Returns
        -------
        means:
            The predicted response of shape `(test_count, response_count,)` for
            each of the test examples.
        variances:
            The independent posterior variances for each of the test examples of
            shape `(test_count,)`.
        """
        test_count = test_features.shape[0]
        test_nn_indices, _ = train_nbrs_lookup.get_nns(test_features)

        pairwise_tensor, crosswise_tensor, batch_nn_targets = tensors_from_indices(
        self, np.arange(test_count), test_nn_indices, test_features, train_features, train_targets
        )

        return self.posterior_mean(
            pairwise_tensor, crosswise_tensor, batch_nn_targets
        ), self.posterior_variance(pairwise_tensor, crosswise_tensor)
    
    def predict_one(
        self,
        test_feature: np.ndarray,
        train_features: np.ndarray,
        train_nbrs_lookup: NN_Wrapper,
        train_targets: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predicts the response for each test item one at a time using training nearest neighbors as reference.  Useful in the context of performing MCMC studies

        Args:
            self:
                Regressor object.
            test_feature:
                Test observation of shape `(1, feature_count)`.
            train_features:
                Train observations of shape `(train_count, feature_count)`.
            train_nbrs_lookup:
                Trained nearest neighbor query data structure.
            train_targets:
                Observed response for all training data of shape
                `(train_count, class_count)`.

        Returns
        -------
        mean:
            The predicted response of shape `(1, response_count,)` for
            the given test example.
        variance:
            The independent posterior variance for the given test example of
            shape `(1,)`.
        """

        test_count = test_feature.shape[0]
        assert test_count == 1, f"Can only have 1 test item, but got {test_count}"
        indices = np.arange(test_count)
        test_nn_indices, _ = train_nbrs_lookup.get_nns(test_feature) 
  

        (
            crosswise_tensor,
            pairwise_tensor,
            batch_nn_targets,
        ) = self.make_predict_tensors(indices, test_nn_indices, test_feature, train_features, train_targets)
        if isinstance(self, MuyGPS):
            pairwise_tensor = self.kernel(pairwise_tensor)
            crosswise_tensor = self.kernel(crosswise_tensor)

        return self.posterior_mean(
            pairwise_tensor, crosswise_tensor, batch_nn_targets
        )[0][0], self.posterior_variance(pairwise_tensor, crosswise_tensor)[0][0][0]