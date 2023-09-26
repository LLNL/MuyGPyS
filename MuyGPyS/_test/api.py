# Copyright 2021-2023 Lawrence Livermore National Security, LLC and other
# MuyGPyS Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: MIT

from typing import Callable, Dict, List, Tuple, Union

from absl.testing import parameterized

import MuyGPyS._src.math.numpy as np
from MuyGPyS._src.mpi_utils import (
    _consistent_chunk_tensor,
    _consistent_unchunk_tensor,
    _consistent_reduce_scalar,
)
from MuyGPyS._test.utils import _check_ndarray
from MuyGPyS.examples.classify import do_classify
from MuyGPyS.examples.two_class_classify_uq import do_classify_uq, do_uq
from MuyGPyS.examples.regress import do_regress
from MuyGPyS.examples.fast_posterior_mean import do_fast_posterior_mean
from MuyGPyS.gp import MuyGPS, MultivariateMuyGPS as MMuyGPS
from MuyGPyS.gp.kernels import Matern
from MuyGPyS.optimize import OptimizeFn
from MuyGPyS.optimize.loss import mse_fn, LossFn


class APITestCase(parameterized.TestCase):
    def _check_ndarray(self, *args, **kwargs):
        return _check_ndarray(self.assertEqual, args, kwargs)


class ClassifyAPITest(APITestCase):
    def _do_classify_test_chassis(
        self,
        train: Dict[str, np.ndarray],
        test: Dict[str, np.ndarray],
        target_acc: float,
        nn_count: int,
        batch_count: int,
        loss_fn: LossFn,
        opt_fn: OptimizeFn,
        nn_kwargs: Dict,
        k_kwargs: Union[Dict, Union[List[Dict], Tuple[Dict, ...]]],
        opt_kwargs: Dict,
        verbose: bool = False,
    ) -> None:
        (
            muygps,
            surrogate_predictions,
            predicted_labels,
            acc,
        ) = self._do_classify(
            train,
            test,
            nn_count,
            batch_count,
            loss_fn,
            opt_fn,
            nn_kwargs,
            k_kwargs,
            opt_kwargs,
            verbose=verbose,
        )
        self.assertEqual(surrogate_predictions.shape, test["output"].shape)
        self.assertEqual(predicted_labels.shape, (test["output"].shape[0],))
        if isinstance(k_kwargs, dict):
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
            param_names, param_vals, _ = muygps.get_opt_params()
            for i, p in enumerate(param_names):
                print(f"\t{p} : {param_vals[i]}")
        elif isinstance(muygps, MMuyGPS):
            for i, model in enumerate(muygps.models):
                print(f"model {i}:")
                param_names, param_vals, _ = model.get_opt_params()
                for i, p in enumerate(param_names):
                    print(f"\t{p} : {param_vals[i]}")
        print(f"obtains accuracy: {acc}")
        self.assertGreaterEqual(acc, target_acc)

    def _do_classify(
        self,
        train: Dict[str, np.ndarray],
        test: Dict[str, np.ndarray],
        nn_count: int,
        batch_count: int,
        loss_fn: LossFn,
        opt_fn: OptimizeFn,
        nn_kwargs: Dict,
        k_kwargs: Union[Dict, Union[List[Dict], Tuple[Dict, ...]]],
        opt_kwargs: Dict,
        verbose: bool = False,
    ) -> Tuple[Union[MuyGPS, MMuyGPS], np.ndarray, np.ndarray, float]:
        classifier, _, surrogate_predictions = do_classify(
            test["input"],
            train["input"],
            train["output"],
            nn_count=nn_count,
            batch_count=batch_count,
            loss_fn=loss_fn,
            opt_fn=opt_fn,
            k_kwargs=k_kwargs,
            nn_kwargs=nn_kwargs,
            opt_kwargs=opt_kwargs,
            verbose=verbose,
        )

        predicted_labels = np.argmax(surrogate_predictions, axis=1)
        target_labels = np.argmax(test["output"], axis=1)
        test_count = len(target_labels)
        if np.all(np.unique(train["output"]) == np.unique([-1.0, 1.0])):
            predicted_labels = 2 * predicted_labels - 1
            target_labels = 2 * target_labels - 1

        target_labels = _consistent_chunk_tensor(target_labels)
        correct_count = np.count_nonzero(predicted_labels == target_labels)

        correct_count = _consistent_reduce_scalar(correct_count)
        acc = correct_count / test_count

        surrogate_predictions = _consistent_unchunk_tensor(
            surrogate_predictions
        )
        predicted_labels = _consistent_unchunk_tensor(predicted_labels)
        return (
            classifier,
            surrogate_predictions,
            predicted_labels,
            acc,
        )

    def _do_classify_uq_test_chassis(
        self,
        train: Dict[str, np.ndarray],
        test: Dict[str, np.ndarray],
        target_acc: float,
        nn_count: int,
        opt_batch_count: int,
        uq_batch_count: int,
        loss_fn: LossFn,
        opt_fn: OptimizeFn,
        uq_objectives: Union[List[Callable], Tuple[Callable, ...]],
        nn_kwargs: Dict,
        k_kwargs: Dict,
        opt_kwargs: Dict,
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
            loss_fn,
            opt_fn,
            uq_objectives,
            nn_kwargs,
            k_kwargs,
            opt_kwargs,
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
        param_names, param_vals, _ = muygps.get_opt_params()
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
        if isinstance(muygps.kernel, Matern):
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
        loss_fn: LossFn,
        opt_fn: OptimizeFn,
        uq_objectives: Union[List[Callable], Tuple[Callable, ...]],
        nn_kwargs: Dict,
        k_kwargs: Dict,
        opt_kwargs: Dict,
        verbose: bool = False,
    ) -> Tuple[MuyGPS, np.ndarray, np.ndarray, np.ndarray, float]:
        muygps, _, surrogate_predictions, masks = do_classify_uq(
            test["input"],
            train["input"],
            train["output"],
            nn_count=nn_count,
            opt_batch_count=opt_batch_count,
            uq_batch_count=uq_batch_count,
            loss_fn=loss_fn,
            opt_fn=opt_fn,
            uq_objectives=uq_objectives,
            k_kwargs=k_kwargs,
            nn_kwargs=nn_kwargs,
            opt_kwargs=opt_kwargs,
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
        loss_fn: LossFn,
        opt_fn: OptimizeFn,
        nn_kwargs: Dict,
        k_kwargs: Union[Dict, Union[List[Dict], Tuple[Dict, ...]]],
        opt_kwargs: Dict,
        verbose: bool = False,
    ) -> None:
        regressor, predictions, mse, variance = self._do_regress(
            train,
            test,
            nn_count,
            batch_count,
            loss_fn,
            opt_fn,
            nn_kwargs,
            k_kwargs,
            opt_kwargs,
            verbose=verbose,
        )
        self.assertEqual(predictions.shape, test["output"].shape)
        if isinstance(regressor, MuyGPS):
            self._verify_regressor(regressor, variance, test["output"])
        else:
            test_count, _ = test["output"].shape
            for i, model in enumerate(regressor.models):
                self._verify_regressor(
                    model,
                    variance[:, i].reshape(test_count, 1),
                    test["output"][:, i].reshape(test_count, 1),
                )
        print(f"obtains mse: {mse}")
        self.assertLessEqual(mse, target_mse)

    def _verify_regressor(self, regressor, variance, targets):
        param_names, param_vals, _ = regressor.get_opt_params()
        if len(param_names) > 0:
            print("finds hyperparameters:")
            for i, p in enumerate(param_names):
                print(f"\t{p} : {param_vals[i]}")
        if variance is not None:
            test_count, response_count = targets.shape
            self.assertEqual(variance.shape, (test_count, response_count))
            if regressor.fixed():
                self.assertFalse(regressor.scale.trained)
            else:
                self.assertTrue(regressor.scale.trained)
            self.assertEqual(regressor.scale.shape, (response_count,))
            _check_ndarray(self.assertEqual, regressor.scale(), np.ftype)

    def _do_regress(
        self,
        train: Dict[str, np.ndarray],
        test: Dict[str, np.ndarray],
        nn_count: int,
        batch_count: int,
        loss_fn: LossFn,
        opt_fn: OptimizeFn,
        nn_kwargs: Dict,
        k_kwargs: Union[Dict, Union[List[Dict], Tuple[Dict, ...]]],
        opt_kwargs: Dict,
        verbose: bool = False,
    ) -> Tuple[Union[MuyGPS, MMuyGPS], np.ndarray, float, np.ndarray,]:
        # print("gets here")
        regressor, _, predictions, variance = do_regress(
            test["input"],
            train["input"],
            train["output"],
            nn_count=nn_count,
            batch_count=batch_count,
            loss_fn=loss_fn,
            opt_fn=opt_fn,
            k_kwargs=k_kwargs,
            nn_kwargs=nn_kwargs,
            opt_kwargs=opt_kwargs,
            verbose=verbose,
        )

        predictions = _consistent_unchunk_tensor(predictions)
        variance = _consistent_unchunk_tensor(variance)
        mse = mse_fn(predictions, test["output"])
        return (regressor, predictions, mse, variance)  # type: ignore


class FastPosteriorMeanAPITest(parameterized.TestCase):
    def _do_fast_posterior_mean_test_chassis(
        self,
        train: Dict[str, np.ndarray],
        test: Dict[str, np.ndarray],
        target_mse: float,
        nn_count: int,
        batch_count: int,
        loss_fn: LossFn,
        opt_fn: OptimizeFn,
        nn_kwargs: Dict,
        k_kwargs: Union[Dict, Union[List[Dict], Tuple[Dict, ...]]],
        opt_kwargs: Dict,
        verbose: bool = False,
    ) -> None:
        regressor, predictions, mse = self._do_fast_posterior_mean(
            train=train,
            test=test,
            nn_count=nn_count,
            batch_count=batch_count,
            loss_fn=loss_fn,
            opt_fn=opt_fn,
            nn_kwargs=nn_kwargs,
            k_kwargs=k_kwargs,
            opt_kwargs=opt_kwargs,
            verbose=verbose,
        )
        self.assertEqual(predictions.shape, test["output"].shape)
        print(f"obtains mse: {mse}")
        self.assertLessEqual(mse, target_mse)

        if isinstance(regressor, MuyGPS):
            param_names, param_vals, _ = regressor.get_opt_params()
            for i, p in enumerate(param_names):
                print(f"\t{p} : {param_vals[i]}")
        elif isinstance(regressor, MMuyGPS):
            for i, model in enumerate(regressor.models):
                print(f"model {i}:")
                param_names, param_vals, _ = model.get_opt_params()
                for i, p in enumerate(param_names):
                    print(f"\t{p} : {param_vals[i]}")

    def _do_fast_posterior_mean(
        self,
        train: Dict[str, np.ndarray],
        test: Dict[str, np.ndarray],
        nn_count: int,
        batch_count: int,
        loss_fn: LossFn,
        opt_fn: OptimizeFn,
        nn_kwargs: Dict,
        k_kwargs: Union[Dict, Union[List[Dict], Tuple[Dict, ...]]],
        opt_kwargs: Dict,
        verbose: bool = False,
    ) -> Tuple[Union[MuyGPS, MMuyGPS], np.ndarray, float]:
        (
            regressor,
            _,
            predictions,
            precomputed_coefficient_matrix,
            _,
        ) = do_fast_posterior_mean(
            test["input"],
            train["input"],
            train["output"],
            nn_count=nn_count,
            batch_count=batch_count,
            loss_fn=loss_fn,
            opt_fn=opt_fn,
            k_kwargs=k_kwargs,
            nn_kwargs=nn_kwargs,
            opt_kwargs=opt_kwargs,
            verbose=verbose,
        )

        mse = mse_fn(predictions, test["output"])
        return (
            regressor,
            predictions,
            mse,  # type: ignore
        )
