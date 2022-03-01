# Copyright 2021 Lawrence Livermore National Security, LLC and other MuyGPyS
# Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: MIT

from MuyGPyS.neighbors import NN_Wrapper
import numpy as np

from typing import cast, Callable, Dict, List, Optional, Tuple, Union

from absl.testing import parameterized

from MuyGPyS.examples.classify import do_classify
from MuyGPyS.examples.two_class_classify_uq import do_classify_uq, do_uq
from MuyGPyS.examples.regress import do_regress
from MuyGPyS.gp.muygps import MuyGPS, MultivariateMuyGPS as MMuyGPS
from MuyGPyS.optimize.objective import mse_fn


class ClassifyAPITest(parameterized.TestCase):
    def _do_classify_test_chassis(
        self,
        train: Dict[str, np.ndarray],
        test: Dict[str, np.ndarray],
        target_acc: float,
        nn_count: int,
        batch_count: int,
        loss_method: str,
        nn_kwargs: Dict,
        k_kwargs: Union[Dict, Union[List[Dict], Tuple[Dict, ...]]],
        kern: Optional[str] = None,
        return_distances: bool = False,
        verbose: bool = False,
    ) -> None:
        (
            muygps,
            surrogate_predictions,
            predicted_labels,
            acc,
            crosswise_dists,
            pairwise_dists,
        ) = self._do_classify(
            train,
            test,
            nn_count,
            batch_count,
            loss_method,
            nn_kwargs,
            kern,
            k_kwargs,
            return_distances=return_distances,
            verbose=verbose,
        )
        self.assertEqual(surrogate_predictions.shape, test["output"].shape)
        self.assertEqual(predicted_labels.shape, (test["output"].shape[0],))
        if kern is None:
            self.assertSequenceAlmostEqual(
                np.sum(surrogate_predictions, axis=1),
                np.zeros(surrogate_predictions.shape[0]),
            )
        # There is almost certainly a better way to do this.
        if np.all(np.unique(train["output"]) == np.unique([-1, 1])):
            self.assertSequenceAlmostEqual(
                np.unique(predicted_labels),
                np.unique(train["output"]),
            )
        else:
            self.assertSequenceAlmostEqual(
                np.unique(predicted_labels),
                np.unique(np.argmax(train["output"], axis=1)),
            )
        print("Finds hyperparameters:")
        if isinstance(muygps, MuyGPS):
            param_names, param_vals, _ = muygps.get_optim_params()
            for i, p in enumerate(param_names):
                print(f"\t{p} : {param_vals[i]}")
        elif isinstance(muygps, MMuyGPS):
            for i, model in enumerate(muygps.models):
                print(f"model {i}:")
                param_names, param_vals, _ = model.get_optim_params()
                for i, p in enumerate(param_names):
                    print(f"\t{p} : {param_vals[i]}")
        print(f"obtains accuracy: {acc}")
        self.assertGreaterEqual(acc, target_acc)
        if crosswise_dists is not None:
            self.assertEqual(crosswise_dists.shape, (batch_count, nn_count))
        if pairwise_dists is not None:
            self.assertEqual(
                pairwise_dists.shape, (batch_count, nn_count, nn_count)
            )

    def _do_classify(
        self,
        train: Dict[str, np.ndarray],
        test: Dict[str, np.ndarray],
        nn_count: int,
        batch_count: int,
        loss_method: str,
        nn_kwargs: Dict,
        kern: Optional[str],
        k_kwargs: Union[Dict, Union[List[Dict], Tuple[Dict, ...]]],
        return_distances: bool = False,
        verbose: bool = False,
    ) -> Tuple[
        Union[MuyGPS, MMuyGPS],
        np.ndarray,
        np.ndarray,
        float,
        np.ndarray,
        np.ndarray,
    ]:
        ret = do_classify(
            test["input"],
            train["input"],
            train["output"],
            nn_count=nn_count,
            batch_count=batch_count,
            loss_method=loss_method,
            kern=kern,
            k_kwargs=k_kwargs,
            nn_kwargs=nn_kwargs,
            return_distances=return_distances,
            verbose=verbose,
        )

        crosswise_dists = None
        pairwise_dists = None
        if return_distances is False:
            classifier, _, surrogate_predictions = cast(
                Tuple[Union[MuyGPS, MMuyGPS], NN_Wrapper, np.ndarray], ret
            )
        else:
            (
                classifier,
                _,
                surrogate_predictions,
                crosswise_dists,
                pairwise_dists,
            ) = cast(
                Tuple[
                    Union[MuyGPS, MMuyGPS],
                    NN_Wrapper,
                    np.ndarray,
                    np.ndarray,
                    np.ndarray,
                ],
                ret,
            )

        predicted_labels = np.argmax(surrogate_predictions, axis=1)
        acc = np.mean(predicted_labels == np.argmax(test["output"], axis=1))
        if np.all(np.unique(train["output"]) == np.unique([-1.0, 1.0])):
            predicted_labels = 2 * predicted_labels - 1
        return (
            classifier,
            surrogate_predictions,
            predicted_labels,
            acc,
            crosswise_dists,
            pairwise_dists,
        )

    def _do_classify_uq_test_chassis(
        self,
        train: Dict[str, np.ndarray],
        test: Dict[str, np.ndarray],
        target_acc: float,
        nn_count: int,
        opt_batch_count: int,
        uq_batch_count: int,
        loss_method: str,
        uq_objectives: Union[List[Callable], Tuple[Callable, ...]],
        nn_kwargs: Dict,
        k_kwargs: Dict,
        verbose: bool = False,
    ) -> None:
        (
            muygps,
            surrogate_predictions,
            predicted_labels,
            masks,
            acc,
        ) = self._do_classify_uq(
            train,
            test,
            nn_count,
            opt_batch_count,
            uq_batch_count,
            loss_method,
            uq_objectives,
            nn_kwargs,
            k_kwargs,
            verbose=verbose,
        )
        self.assertEqual(surrogate_predictions.shape, test["output"].shape)
        self.assertEqual(predicted_labels.shape, (test["output"].shape[0],))
        self.assertSequenceAlmostEqual(
            np.sum(surrogate_predictions, axis=1),
            np.zeros(surrogate_predictions.shape[0]),
        )
        # There is almost certainly a better way to do this.
        if np.all(np.unique(train["output"]) == np.unique([-1, 1])):
            self.assertSequenceAlmostEqual(
                np.unique(predicted_labels),
                np.unique(train["output"]),
            )
        else:
            self.assertSequenceAlmostEqual(
                np.unique(predicted_labels),
                np.unique(np.argmax(train["output"], axis=1)),
            )
        print("Finds hyperparameters:")
        param_names, param_vals, _ = muygps.get_optim_params()
        for i, p in enumerate(param_names):
            print(f"\t{p} : {param_vals[i]}")
        print(f"obtains accuracy: {acc}")
        self.assertGreaterEqual(acc, target_acc)
        accuracy, uq = do_uq(surrogate_predictions, test["output"], masks)
        self.assertEqual(
            masks.shape, (len(uq_objectives), test["output"].shape[0])
        )
        self.assertEqual(uq.shape, (len(uq_objectives), 3))
        # @NOTE[bwp] Should we do more to validate the uq? Expected ranges?
        # What about the first dimension `np.sum(mask)`, which records the
        # number of "ambiguous" prediction locations?
        # print(uq)
        if muygps.kern == "matern":
            for i in range(uq.shape[0] - 1):
                self.assertLessEqual(uq[i, 1], acc)
                self.assertGreaterEqual(uq[i, 2], acc)
        self.assertLessEqual(uq[-1, 1], 1.0)
        self.assertGreaterEqual(uq[-1, 2], target_acc)

    def _do_classify_uq(
        self,
        train: Dict[str, np.ndarray],
        test: Dict[str, np.ndarray],
        nn_count: int,
        opt_batch_count: int,
        uq_batch_count: int,
        loss_method: str,
        uq_objectives: Union[List[Callable], Tuple[Callable, ...]],
        nn_kwargs: Dict,
        k_kwargs: Dict,
        verbose: bool = False,
    ) -> Tuple[MuyGPS, np.ndarray, np.ndarray, np.ndarray, float]:
        muygps, _, surrogate_predictions, masks = do_classify_uq(
            test["input"],
            train["input"],
            train["output"],
            nn_count=nn_count,
            opt_batch_count=opt_batch_count,
            uq_batch_count=uq_batch_count,
            loss_method=loss_method,
            uq_objectives=uq_objectives,
            k_kwargs=k_kwargs,
            nn_kwargs=nn_kwargs,
            verbose=verbose,
        )

        predicted_labels = np.argmax(surrogate_predictions, axis=1)
        acc = np.mean(predicted_labels == np.argmax(test["output"], axis=1))
        if np.all(np.unique(train["output"]) == np.unique([-1.0, 1.0])):
            predicted_labels = 2 * predicted_labels - 1
        return muygps, surrogate_predictions, predicted_labels, masks, acc


class RegressionAPITest(parameterized.TestCase):
    def _do_regress_test_chassis(
        self,
        train: Dict[str, np.ndarray],
        test: Dict[str, np.ndarray],
        target_mse: float,
        nn_count: int,
        batch_count: int,
        loss_method: str,
        sigma_method: Optional[str],
        variance_mode: Optional[str],
        nn_kwargs: Dict,
        k_kwargs: Union[Dict, Union[List[Dict], Tuple[Dict, ...]]],
        kern: Optional[str] = None,
        apply_sigma_sq: bool = False,
        return_distances: bool = False,
        verbose: bool = False,
    ) -> None:
        (
            regressor,
            predictions,
            mse,
            variance,
            crosswise_dists,
            pairwise_dists,
        ) = self._do_regress(
            train,
            test,
            nn_count,
            batch_count,
            loss_method,
            sigma_method,
            variance_mode,
            nn_kwargs,
            k_kwargs,
            kern=kern,
            apply_sigma_sq=apply_sigma_sq,
            return_distances=return_distances,
            verbose=verbose,
        )
        self.assertEqual(predictions.shape, test["output"].shape)
        if isinstance(regressor, MuyGPS):
            self._verify_regressor(
                regressor, variance, test["output"], sigma_method
            )
        else:
            test_count, _ = test["output"].shape
            for i, model in enumerate(regressor.models):
                self._verify_regressor(
                    model,
                    variance[:, i] if variance is not None else None,
                    test["output"][:, i].reshape(test_count, 1),
                    sigma_method,
                )
        print(f"obtains mse: {mse}")
        self.assertLessEqual(mse, target_mse)
        if crosswise_dists is not None:
            self.assertEqual(crosswise_dists.shape, (batch_count, nn_count))
        if pairwise_dists is not None:
            self.assertEqual(
                pairwise_dists.shape, (batch_count, nn_count, nn_count)
            )

    def _verify_regressor(self, regressor, variance, targets, sigma_method):
        param_names, param_vals, _ = regressor.get_optim_params()
        if len(param_names) > 0:
            print("finds hyperparameters:")
            for i, p in enumerate(param_names):
                print(f"\t{p} : {param_vals[i]}")
        if variance is not None:
            test_count, response_count = targets.shape
            if response_count > 1:
                self.assertEqual(variance.shape, (test_count, response_count))
            else:
                self.assertEqual(variance.shape, (test_count,))
        if sigma_method is None:
            self.assertFalse(regressor.sigma_sq.trained())
        elif sigma_method.lower() == "analytic":
            self.assertEqual(regressor.sigma_sq().shape, (response_count,))
            self.assertEqual(regressor.sigma_sq().dtype, float)
        else:
            raise ValueError(f"Unsupported sigma method {sigma_method}")

    def _do_regress(
        self,
        train: Dict[str, np.ndarray],
        test: Dict[str, np.ndarray],
        nn_count: int,
        batch_count: int,
        loss_method: str,
        sigma_method: Optional[str],
        variance_mode: Optional[str],
        nn_kwargs: Dict,
        k_kwargs: Union[Dict, Union[List[Dict], Tuple[Dict, ...]]],
        kern: Optional[str] = None,
        apply_sigma_sq: bool = True,
        return_distances: bool = False,
        verbose: bool = False,
    ) -> Tuple[
        Union[MuyGPS, MMuyGPS],
        np.ndarray,
        float,
        np.ndarray,
        np.ndarray,
        np.ndarray,
    ]:
        ret = do_regress(
            test["input"],
            train["input"],
            train["output"],
            nn_count=nn_count,
            batch_count=batch_count,
            loss_method=loss_method,
            sigma_method=sigma_method,
            variance_mode=variance_mode,
            kern=kern,
            k_kwargs=k_kwargs,
            nn_kwargs=nn_kwargs,
            apply_sigma_sq=apply_sigma_sq,
            return_distances=return_distances,
            verbose=verbose,
        )
        variance = None
        crosswise_dists = None
        pairwise_dists = None
        if variance_mode is None and return_distances is False:
            regressor, _, predictions = cast(
                Tuple[Union[MuyGPS, MMuyGPS], NN_Wrapper, np.ndarray], ret
            )
        elif variance_mode == "diagonal":
            if return_distances is False:
                regressor, _, predictions, variance = cast(
                    Tuple[
                        Union[MuyGPS, MMuyGPS],
                        NN_Wrapper,
                        np.ndarray,
                        np.ndarray,
                    ],
                    ret,
                )
            else:
                (
                    regressor,
                    _,
                    predictions,
                    variance,
                    crosswise_dists,
                    pairwise_dists,
                ) = cast(
                    Tuple[
                        Union[MuyGPS, MMuyGPS],
                        NN_Wrapper,
                        np.ndarray,
                        np.ndarray,
                        np.ndarray,
                        np.ndarray,
                    ],
                    ret,
                )
        else:
            raise ValueError(f"Variance mode {variance_mode} is not supported.")
        mse = mse_fn(predictions, test["output"])
        return (
            regressor,
            predictions,
            mse,
            variance,
            crosswise_dists,
            pairwise_dists,
        )
