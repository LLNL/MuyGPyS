# Copyright 2021 Lawrence Livermore National Security, LLC and other MuyGPyS
# Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: (Apache-2.0 OR MIT)

import numpy as np
from absl.testing import parameterized

from MuyGPyS.examples.classify import (
    do_classify,
    do_classify_uq,
    do_uq,
)
from MuyGPyS.examples.regress import do_regress
from MuyGPyS.gp.muygps import MuyGPS, MultivariateMuyGPS as MMuyGPS
from MuyGPyS.optimize.objective import mse_fn


class ClassifyAPITest(parameterized.TestCase):
    def _do_classify_test_chassis(
        self,
        train,
        test,
        target_acc,
        nn_count,
        batch_size,
        loss_method,
        nn_kwargs,
        k_kwargs,
        kern=None,
        verbose=False,
    ):
        (
            muygps,
            surrogate_predictions,
            predicted_labels,
            acc,
        ) = self._do_classify(
            train,
            test,
            nn_count,
            batch_size,
            loss_method,
            nn_kwargs,
            kern,
            k_kwargs,
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
        print(f"Finds hyperparameters:")
        if isinstance(muygps, MuyGPS):
            optim_params = muygps.get_optim_params()
            for p in optim_params:
                print(f"\t{p} : {optim_params[p]()}")
        elif isinstance(muygps, MMuyGPS):
            for i, model in enumerate(muygps.models):
                print(f"model {i}:")
                optim_params = model.get_optim_params()
                for p in optim_params:
                    print(f"\t{p} : {optim_params[p]()}")
        print(f"obtains accuracy: {acc}")
        self.assertGreaterEqual(acc, target_acc)

    def _do_classify(
        self,
        train,
        test,
        nn_count,
        batch_size,
        loss_method,
        nn_kwargs,
        kern,
        k_kwargs,
        verbose=False,
    ):
        muygps, _, surrogate_predictions = do_classify(
            test["input"],
            train["input"],
            train["output"],
            nn_count=nn_count,
            batch_size=batch_size,
            loss_method=loss_method,
            kern=kern,
            k_kwargs=k_kwargs,
            nn_kwargs=nn_kwargs,
            verbose=verbose,
        )

        predicted_labels = np.argmax(surrogate_predictions, axis=1)
        acc = np.mean(predicted_labels == np.argmax(test["output"], axis=1))
        if np.all(np.unique(train["output"]) == np.unique([-1.0, 1.0])):
            predicted_labels = 2 * predicted_labels - 1
        return muygps, surrogate_predictions, predicted_labels, acc

    def _do_classify_uq_test_chassis(
        self,
        train,
        test,
        target_acc,
        nn_count,
        opt_batch_size,
        uq_batch_size,
        loss_method,
        uq_objectives,
        nn_kwargs,
        k_kwargs,
        verbose=False,
    ):
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
            opt_batch_size,
            uq_batch_size,
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
        print(f"Finds hyperparameters:")
        optim_params = muygps.get_optim_params()
        for p in optim_params:
            print(f"\t{p} : {optim_params[p]()}")
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
        train,
        test,
        nn_count,
        opt_batch_size,
        uq_batch_size,
        loss_method,
        uq_objectives,
        nn_kwargs,
        k_kwargs,
        verbose=False,
    ):
        muygps, _, surrogate_predictions, masks = do_classify_uq(
            test["input"],
            train["input"],
            train["output"],
            nn_count=nn_count,
            opt_batch_size=opt_batch_size,
            uq_batch_size=uq_batch_size,
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

    def _old_do_classify_test_chassis(
        self,
        train,
        test,
        target_acc,
        nn_count,
        embed_dim,
        opt_batch_size,
        uq_batch_size,
        loss_method,
        embed_method,
        uq_objectives,
        nn_kwargs,
        kern,
        hyper_dict,
    ):
        surrogate_predictions, predicted_labels, acc, masks = self._do_classify(
            train=train,
            test=test,
            nn_count=nn_count,
            embed_dim=embed_dim,
            opt_batch_size=opt_batch_size,
            uq_batch_size=uq_batch_size,
            loss_method=loss_method,
            embed_method=embed_method,
            uq_objectives=uq_objectives,
            nn_kwargs=nn_kwargs,
            kern=kern,
            hyper_dict=hyper_dict,
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
        self.assertGreaterEqual(acc, target_acc)
        if masks is not None:
            accuracy, uq = do_uq(surrogate_predictions, test["output"], masks)
            self.assertEqual(
                masks.shape, (len(uq_objectives), test["output"].shape[0])
            )
            self.assertEqual(uq.shape, (len(uq_objectives), 3))
            # @NOTE[bwp] Should we do more to validate the uq? Expected ranges?
            # What about the first dimension `np.sum(mask)`, which records the
            # number of "ambiguous" prediction locations?
            # print(uq)
            if kern == "matern":
                for i in range(uq.shape[0] - 1):
                    self.assertLessEqual(uq[i, 1], acc)
                    self.assertGreaterEqual(uq[i, 2], acc)
            self.assertLessEqual(uq[-1, 1], 1.0)
            self.assertGreaterEqual(uq[-1, 2], target_acc)

    def _old_do_classify(
        self,
        train,
        test,
        nn_count,
        embed_dim,
        opt_batch_size,
        uq_batch_size,
        loss_method,
        embed_method,
        uq_objectives,
        nn_kwargs,
        kern,
        hyper_dict,
    ):
        surrogate_predictions = do_classify(
            train,
            test,
            embed_method=embed_method,
            opt_batch_size=opt_batch_size,
            uq_batch_size=uq_batch_size,
            embed_dim=embed_dim,
            nn_count=nn_count,
            kern=kern,
            hyper_dict=hyper_dict,
            uq_objectives=uq_objectives,
            nn_kwargs=nn_kwargs,
            verbose=False,
        )
        if uq_objectives is not None:
            surrogate_predictions, masks = surrogate_predictions
        else:
            masks = None
        predicted_labels = np.argmax(surrogate_predictions, axis=1)
        acc = np.mean(predicted_labels == np.argmax(test["output"], axis=1))
        if np.all(np.unique(train["output"]) == np.unique([-1, 1])):
            predicted_labels = 2 * predicted_labels - 1
        return surrogate_predictions, predicted_labels, acc, masks


class RegressionAPITest(parameterized.TestCase):
    def _do_regress_test_chassis(
        self,
        train,
        test,
        target_mse,
        nn_count,
        # embed_dim,
        batch_size,
        loss_method,
        variance_mode,
        # embed_method,
        nn_kwargs,
        k_kwargs,
        kern=None,
        verbose=False,
    ):
        regressor, predictions, mse, variance = self._do_regress(
            train,
            test,
            nn_count,
            # embed_dim,
            batch_size,
            loss_method,
            variance_mode,
            # embed_method,
            nn_kwargs,
            k_kwargs,
            kern=kern,
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
                    variance[:, i] if variance is not None else None,
                    test["output"][:, i].reshape(test_count, 1),
                )
        print(f"obtains mse: {mse}")
        self.assertLessEqual(mse, target_mse)

    def _verify_regressor(self, regressor, variance, targets):
        optim_params = regressor.get_optim_params()
        if len(optim_params) > 0:
            print(f"finds hyperparameters:")
            for p in optim_params:
                print(f"\t{p} : {optim_params[p]()}")
        if variance is not None:
            test_count, response_count = targets.shape
            self.assertEqual(variance.shape, (test_count,))
            if response_count > 1:
                self.assertEqual(regressor.sigma_sq().shape, (response_count,))

    def _do_regress(
        self,
        train,
        test,
        nn_count,
        batch_size,
        loss_method,
        variance_mode,
        nn_kwargs,
        k_kwargs,
        kern=None,
        verbose=False,
    ):
        ret = do_regress(
            test["input"],
            train["input"],
            train["output"],
            nn_count=nn_count,
            batch_size=batch_size,
            loss_method=loss_method,
            variance_mode=variance_mode,
            kern=kern,
            k_kwargs=k_kwargs,
            nn_kwargs=nn_kwargs,
            verbose=verbose,
        )
        if variance_mode is None:
            regressor, _, predictions = ret
            variance = None
            # sigma_sq = None
        elif variance_mode == "diagonal":
            regressor, _, predictions, variance = ret
            # predictions, variance = predictions
            # sigma_sq = (
            #     muygps.sigma_sq()
            # )  # np.array([ss() for ss in muygps.sigma_sq])
        else:
            raise ValueError(f"Variance mode {variance_mode} is not supported.")
        mse = mse_fn(predictions, test["output"])
        return regressor, predictions, mse, variance