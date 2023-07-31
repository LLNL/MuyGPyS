# Copyright 2021-2023 Lawrence Livermore National Security, LLC and other
# MuyGPyS Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: MIT


import numpy as np
import matplotlib.pyplot as plt

from MuyGPyS._test.gp import benchmark_sample, BenchmarkGP
from MuyGPyS.gp.distortion import IsotropicDistortion
from MuyGPyS.gp.hyperparameter import ScalarHyperparameter
from MuyGPyS.gp.kernels import Matern
from MuyGPyS.gp.noise import HomoscedasticNoise
from MuyGPyS.optimize.loss import mse_fn


class SamplerBase:
    def _train_mask(self):
        return slice(None, None, self.train_step)

    def _test_mask(self):
        return np.mod(np.arange(self.data_count), self.train_step) != 0


class UnivariateSampler(SamplerBase):
    def __init__(
        self,
        data_count=5001,
        train_step=10,
        kernel=Matern(
            nu=ScalarHyperparameter(2.0),
            metric=IsotropicDistortion(
                "l2", length_scale=ScalarHyperparameter(1.0)
            ),
        ),
        eps=HomoscedasticNoise(1e-14),
        measurement_eps=HomoscedasticNoise(1e-5),
    ):
        self.data_count = data_count
        self.train_step = train_step
        self.x = np.linspace(0, 1, data_count).reshape(data_count, 1)
        self.test_features = self.x[self._test_mask(), :]
        self.train_features = self.x[self._train_mask(), :]
        self.test_count, _ = self.test_features.shape
        self.train_count, _ = self.train_features.shape
        self.measurement_eps = measurement_eps
        self.gp = BenchmarkGP(kernel=kernel, eps=eps)
        self.vis_subset_size = 10
        self.mid = int(self.train_count / 2)

    def features(self):
        return self.train_features, self.test_features

    def sample(self):
        y = benchmark_sample(self.gp, self.x)
        self.test_responses = y[self._test_mask(), :]
        self.train_responses = y[self._train_mask(), :] + np.random.normal(
            0, self.measurement_eps(), size=(self.train_count, 1)
        )
        return self.train_responses, self.test_responses

    def plot_sample(self):
        _, axes = plt.subplots(2, 1, figsize=(15, 11))

        axes[0].set_title("Sampled Curve", fontsize=24)
        axes[0].set_xlabel("Feature Domain", fontsize=20)
        axes[0].set_ylabel("Response Range", fontsize=20)
        axes[0].plot(
            self.train_features,
            self.train_responses,
            "k*",
            label="perturbed train response",
        )
        axes[0].plot(
            self.test_features,
            self.test_responses,
            "g-",
            label="test response",
        )
        axes[0].legend(fontsize=20)

        self.plot_target_sub(axes[1])

        plt.tight_layout()

        plt.show()

    def plot_results(
        self,
        scipy_predictions,
        scipy_confidence_intervals,
        bayes_predictions,
        bayes_confidence_intervals,
    ):
        _, axes = plt.subplots(3, 1, figsize=(15, 16))

        self.plot_target(axes[0])
        self.plot_model(
            axes[0], "Bayes", bayes_predictions, bayes_confidence_intervals
        )
        self.plot_target_sub(axes[1])
        self.plot_model_sub(
            axes[1], "scipy", scipy_predictions, scipy_confidence_intervals
        )
        self.plot_target_sub(axes[2])
        self.plot_model_sub(
            axes[2], "Bayes", bayes_predictions, bayes_confidence_intervals
        )

        plt.tight_layout()

        plt.show()

    def plot_target(self, ax):
        ax.set_title("Sampled Curve", fontsize=24)
        ax.set_xlabel("Feature Domain", fontsize=20)
        ax.set_ylabel("Response Range", fontsize=20)
        ax.plot(
            self.train_features,
            self.train_responses,
            "k*",
            label="perturbed train response",
        )
        ax.plot(
            self.test_features,
            self.test_responses,
            "g-",
            label="test response",
        )

    def plot_model(self, ax, name, predictions, confidence_intervals):
        confidence_intervals = confidence_intervals.reshape(self.test_count)
        ax.set_title(f"Sampled Curve with {name} optimized model", fontsize=24)
        ax.plot(
            self.test_features,
            predictions,
            "r--",
            label="test predictions",
        )
        ax.fill_between(
            self.test_features[:, 0],
            (predictions[:, 0] - confidence_intervals),
            (predictions[:, 0] + confidence_intervals),
            facecolor="red",
            alpha=0.25,
            label="95% Confidence Interval",
        )
        ax.legend(fontsize=20)

    def plot_target_sub(self, ax):
        ax.set_title("Sampled Curve (subset)", fontsize=24)
        ax.set_xlabel("Feature Domain", fontsize=20)
        ax.set_ylabel("Response Range", fontsize=20)
        ax.plot(
            self.train_features[self.mid : self.mid + self.vis_subset_size],
            self.train_responses[self.mid : self.mid + self.vis_subset_size],
            "k*",
            label="perturbed train response",
        )
        ax.plot(
            self.test_features[
                self.mid
                * (self.train_step - 1) : self.mid
                * (self.train_step - 1)
                + (self.vis_subset_size * (self.train_step - 1))
            ],
            self.test_responses[
                self.mid
                * (self.train_step - 1) : self.mid
                * (self.train_step - 1)
                + (self.vis_subset_size * (self.train_step - 1))
            ],
            "g-",
            label="test response",
        )

    def plot_model_sub(self, ax, name, predictions, confidence_intervals):
        confidence_intervals = confidence_intervals.reshape(self.test_count)
        ax.set_title(
            f"Sampled Curve (subset) with {name} optimized model", fontsize=24
        )
        ax.plot(
            self.test_features[
                self.mid
                * (self.train_step - 1) : self.mid
                * (self.train_step - 1)
                + (self.vis_subset_size * (self.train_step - 1))
            ],
            predictions[
                self.mid
                * (self.train_step - 1) : self.mid
                * (self.train_step - 1)
                + (self.vis_subset_size * (self.train_step - 1))
            ],
            "b--",
            label="test predictions",
        )
        ax.fill_between(
            self.test_features[
                self.mid
                * (self.train_step - 1) : self.mid
                * (self.train_step - 1)
                + (self.vis_subset_size * (self.train_step - 1))
            ][:, 0],
            (predictions[:, 0] - confidence_intervals)[
                self.mid
                * (self.train_step - 1) : self.mid
                * (self.train_step - 1)
                + (self.vis_subset_size * (self.train_step - 1))
            ],
            (predictions[:, 0] + confidence_intervals)[
                self.mid
                * (self.train_step - 1) : self.mid
                * (self.train_step - 1)
                + (self.vis_subset_size * (self.train_step - 1))
            ],
            facecolor="blue",
            alpha=0.25,
            label="95% Confidence Interval",
        )
        ax.legend(fontsize=20)


class UnivariateSampler2D(SamplerBase):
    def __init__(
        self,
        points_per_dim=5001,
        train_step=10,
        kernel=Matern(
            nu=ScalarHyperparameter(2.0),
            metric=IsotropicDistortion(
                "l2", length_scale=ScalarHyperparameter(1.0)
            ),
        ),
        eps=HomoscedasticNoise(1e-14),
        measurement_eps=HomoscedasticNoise(1e-5),
    ):
        self.points_per_dim = points_per_dim
        self.data_count = self.points_per_dim**2
        self.train_step = train_step
        x = np.linspace(0, 1, points_per_dim)
        xx, yy = np.meshgrid(x, x)
        self.xs = np.array(
            [
                [xx[i, j], yy[i, j]]
                for i in range(points_per_dim)
                for j in range(points_per_dim)
            ]
        )
        self.test_features = self.xs[self._test_mask(), :]
        self.train_features = self.xs[self._train_mask(), :]
        self.test_count, _ = self.test_features.shape
        self.train_count, _ = self.train_features.shape
        self.measurement_eps = measurement_eps
        self.gp = BenchmarkGP(kernel=kernel, eps=eps)
        self.vis_subset_size = 10
        self.mid = int(self.train_count / 2)

    def features(self):
        return self.train_features, self.test_features

    def sample(self):
        self.ys = benchmark_sample(self.gp, self.xs)
        self.test_responses = self.ys[self._test_mask(), :]
        self.train_responses = self.ys[
            self._train_mask(), :
        ] + np.random.normal(
            0, self.measurement_eps(), size=(self.train_count, 1)
        )
        return self.train_responses, self.test_responses

    def plot_sample(self):
        fig, axes = plt.subplots(1, 3, figsize=(19, 4))

        vmin = np.nanmin(self.ys)
        vmax = np.nanmax(self.ys)

        axes[0].set_title("Sampled Surface", fontsize=24)
        axes[0].set_xlabel("Axis 0", fontsize=20)
        axes[0].set_ylabel("Axis 1", fontsize=20)
        im0 = axes[0].imshow(self._make_im(self.ys), vmin=vmin, vmax=vmax)

        axes[1].set_title("Training Points", fontsize=24)
        axes[1].set_xlabel("Axis 0", fontsize=20)
        axes[1].set_ylabel("Axis 1", fontsize=20)
        train_im = np.zeros(self.data_count)
        train_im[self._train_mask(), None] = self.train_responses
        train_im[self._test_mask()] = -np.inf
        axes[1].imshow(self._make_im(train_im), vmin=vmin, vmax=vmax)

        axes[2].set_title("Testing Points", fontsize=24)
        axes[2].set_xlabel("Axis 0", fontsize=20)
        axes[2].set_ylabel("Axis 1", fontsize=20)
        test_im = np.zeros(self.data_count)
        test_im[self._test_mask(), None] = self.test_responses
        test_im[self._train_mask()] = -np.inf
        axes[2].imshow(self._make_im(test_im), vmin=vmin, vmax=vmax)
        fig.colorbar(im0, ax=axes.ravel().tolist())

        plt.show()

    def _make_im(self, array):
        return array.reshape(self.points_per_dim, self.points_per_dim)

    def plot_predictions(self, predictions):
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))

        test_im = np.zeros(self.data_count)
        test_im[self._test_mask(), None] = self.test_responses

        pred_im = np.zeros(self.data_count)
        pred_im[self._test_mask(), None] = predictions

        vmin = np.min([np.min(test_im), np.min(pred_im)])
        vmax = np.max([np.max(test_im), np.max(pred_im)])

        test_im[self._train_mask()] = -np.inf
        pred_im[self._train_mask()] = -np.inf

        axes[0].set_title("Expected Surface", fontsize=24)
        axes[0].set_xlabel("Axis 0", fontsize=20)
        axes[0].set_ylabel("Axis 1", fontsize=20)
        im0 = axes[0].imshow(
            self._make_im(test_im), vmin=vmin, vmax=vmax, cmap="viridis"
        )
        fig.colorbar(im0, ax=axes[0])

        axes[1].set_title("Predicted Surface", fontsize=24)
        axes[1].set_xlabel("Axis 0", fontsize=20)
        axes[1].set_ylabel("Axis 1", fontsize=20)
        im1 = axes[1].imshow(
            self._make_im(pred_im), vmin=vmin, vmax=vmax, cmap="viridis"
        )
        fig.colorbar(im1, ax=axes[1])

        plt.tight_layout()
        plt.show()

    def plot_error(
        self,
        predictions,
        confidence_intervals,
    ):
        fig, axes = plt.subplots(1, 3, figsize=(13, 4))

        test_im = np.zeros(self.data_count)
        test_im[self._test_mask(), None] = self.test_responses

        pred_im = np.zeros(self.data_count)
        pred_im[self._test_mask(), None] = predictions

        vmin = np.nanmin([np.nanmin(test_im), np.nanmin(pred_im)])
        vmax = np.nanmax([np.nanmax(test_im), np.nanmax(pred_im)])

        resl_im = test_im - pred_im

        conf_im = np.zeros(self.data_count)
        conf_im[self._test_mask(), None] = confidence_intervals

        covr_im = np.zeros(self.data_count)
        covr_im = np.abs(resl_im) - conf_im

        covr_mag = np.max([np.abs(np.max(covr_im)), np.abs(np.min(covr_im))])
        # for i in range(self.data_count):
        #     if np.isnan(conf_im[i]):
        #         continue
        #     if conf_im[i] > resl_im[i]:
        #         update = 1.0
        #     else:
        #         update = -1.0
        #     covr_im[i] = update

        test_im[self._train_mask()] = -np.inf
        pred_im[self._train_mask()] = -np.inf
        resl_im[self._train_mask()] = -np.inf
        conf_im[self._train_mask()] = -np.inf
        covr_im[self._train_mask()] = -np.inf

        axes[0].set_title("Residual", fontsize=24)
        axes[0].set_xlabel("Axis 0", fontsize=20)
        axes[0].set_ylabel("Axis 1", fontsize=20)
        im0 = axes[0].imshow(self._make_im(resl_im), cmap="coolwarm")
        fig.colorbar(im0, ax=axes[0])

        axes[1].set_title("CI Magnitude", fontsize=24)
        axes[1].set_xlabel("Axis 0", fontsize=20)
        axes[1].set_ylabel("Axis 1", fontsize=20)
        im1 = axes[1].imshow(self._make_im(conf_im), cmap="inferno")
        fig.colorbar(im1, ax=axes[1])

        axes[2].set_title("|Residual| - CI", fontsize=24)
        axes[2].set_xlabel("Axis 0", fontsize=20)
        axes[2].set_ylabel("Axis 1", fontsize=20)
        im2 = axes[2].imshow(
            self._make_im(covr_im),
            vmin=-covr_mag,
            vmax=covr_mag,
            cmap="coolwarm",
        )
        cbar2 = fig.colorbar(im2, ax=axes[2])

        plt.tight_layout()
        plt.show()


def print_results(
    name, targets, means, variances, confidence_intervals, coverage
):
    print(f"{name} results:")
    print(f"\tRMSE: {np.sqrt(mse_fn(means, targets))}")
    print(f"\tmean diagonal variance: {np.mean(variances)}")
    print(
        f"\tmean confidence interval size: {np.mean(confidence_intervals * 2)}"
    )
    print(f"\tcoverage: {coverage}")
