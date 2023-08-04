# Copyright 2021-2023 Lawrence Livermore National Security, LLC and other
# MuyGPyS Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: MIT


import numpy as np
import matplotlib.pyplot as plt

from scipy.stats.qmc import LatinHypercube

from MuyGPyS._test.gp import benchmark_sample, BenchmarkGP
from MuyGPyS.gp.distortion import IsotropicDistortion
from MuyGPyS.gp.hyperparameter import ScalarHyperparameter
from MuyGPyS.gp.kernels import Matern
from MuyGPyS.gp.noise import HomoscedasticNoise
from MuyGPyS.optimize.loss import mse_fn


class SamplerBase:
    def __init__(self, data_count, train_ratio=0.1, seed=0):
        rng = np.random.default_rng(seed=seed)
        interval_count = int(data_count * train_ratio)
        interval = int(data_count / interval_count)
        sfl = rng.permutation(np.arange(data_count))
        self._train_mask = np.zeros(data_count, dtype=bool)
        for i in range(interval_count):
            idx = np.random.choice(sfl[i * interval : (i + 1) * interval])
            self._train_mask[idx] = True
        self._test_mask = np.invert(self._train_mask)


class UnivariateSampler(SamplerBase):
    def __init__(
        self,
        data_count=5001,
        train_ratio=0.1,
        seed=0,
        view_lb=0.5,
        view_ub=0.6,
        kernel=Matern(
            nu=ScalarHyperparameter(2.0),
            metric=IsotropicDistortion(
                "l2", length_scale=ScalarHyperparameter(1.0)
            ),
        ),
        eps=HomoscedasticNoise(1e-14),
        measurement_eps=HomoscedasticNoise(1e-5),
    ):
        super().__init__(data_count, train_ratio=train_ratio, seed=seed)
        self.data_count = data_count
        self.x = np.linspace(0, 1, data_count).reshape(data_count, 1)
        self.test_features = self.x[self._test_mask, :]
        self.train_features = self.x[self._train_mask, :]
        self.test_count, _ = self.test_features.shape
        self.train_count, _ = self.train_features.shape
        self.measurement_eps = measurement_eps
        self.gp = BenchmarkGP(kernel=kernel, eps=eps)
        self.train_interval = self.get_interval(
            view_lb, view_ub, self.train_features
        )
        self.test_interval = self.get_interval(
            view_lb, view_ub, self.test_features
        )

    def features(self):
        return self.train_features, self.test_features

    def sample(self):
        y = benchmark_sample(self.gp, self.x)
        self.test_responses = y[self._test_mask, :]
        self.train_responses = y[self._train_mask, :] + np.random.normal(
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

    def get_interval(self, lb, ub, ys):
        return np.array(
            [i for i, x in enumerate(ys) if x >= lb and x <= ub], dtype=int
        )

    def plot_target_sub(self, ax):
        ax.set_title("Sampled Curve (subset)", fontsize=24)
        ax.set_xlabel("Feature Domain", fontsize=20)
        ax.set_ylabel("Response Range", fontsize=20)
        ax.plot(
            self.train_features[self.train_interval],
            self.train_responses[self.train_interval],
            "k*",
            label="perturbed train response",
        )
        ax.plot(
            self.test_features[self.test_interval],
            self.test_responses[self.test_interval],
            "g-",
            label="test response",
        )

    def plot_model_sub(self, ax, name, predictions, confidence_intervals):
        confidence_intervals = confidence_intervals.reshape(self.test_count)
        ax.set_title(
            f"Sampled Curve (subset) with {name} optimized model", fontsize=24
        )
        ax.plot(
            self.test_features[self.test_interval],
            predictions[self.test_interval],
            "b--",
            label="test predictions",
        )
        ax.fill_between(
            self.test_features[self.test_interval][:, 0],
            (predictions[:, 0] - confidence_intervals)[self.test_interval],
            (predictions[:, 0] + confidence_intervals)[self.test_interval],
            facecolor="blue",
            alpha=0.25,
            label="95% Confidence Interval",
        )
        ax.legend(fontsize=20)


class UnivariateSampler2D(SamplerBase):
    def __init__(
        self,
        points_per_dim=60,
        train_ratio=10,
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
        super().__init__(self.data_count, train_ratio)
        x = np.linspace(0, 1, points_per_dim)
        xx, yy = np.meshgrid(x, x)
        self.xs = np.array(
            [
                [xx[i, j], yy[i, j]]
                for i in range(points_per_dim)
                for j in range(points_per_dim)
            ]
        )
        self.test_features = self.xs[self._test_mask, :]
        self.train_features = self.xs[self._train_mask, :]
        self.test_count, _ = self.test_features.shape
        self.train_count, _ = self.train_features.shape
        self.measurement_eps = measurement_eps
        self.gp = BenchmarkGP(kernel=kernel, eps=eps)

    def features(self):
        return self.train_features, self.test_features

    def sample(self):
        self.ys = benchmark_sample(self.gp, self.xs)
        self.test_responses = self.ys[self._test_mask, :]
        self.train_responses = self.ys[self._train_mask, :] + np.random.normal(
            0, self.measurement_eps(), size=(self.train_count, 1)
        )
        return self.train_responses, self.test_responses

    def plot_sample(self):
        fig, axes = plt.subplots(1, 3, figsize=(19, 4))

        vmin = np.nanmin(self.ys)
        vmax = np.nanmax(self.ys)

        self._label_ax(axes[0], "Sampled Surface")
        im0 = axes[0].imshow(
            self._make_im(self.ys, mode="all"), vmin=vmin, vmax=vmax
        )

        self._label_ax(axes[1], "Training Points")
        axes[1].imshow(
            self._make_im(self.train_responses, mode="train"),
            vmin=vmin,
            vmax=vmax,
        )

        self._label_ax(axes[2], "Testing Points")
        axes[2].imshow(self._make_im(self.test_responses), vmin=vmin, vmax=vmax)
        fig.colorbar(im0, ax=axes.ravel().tolist())

        plt.show()

    def _make_im(self, array, mode="test", range=False, add_inf=True):
        im = np.zeros(self.data_count)
        if mode == "test":
            im[self._test_mask, None] = array
            if add_inf is True:
                im[self._train_mask] = -np.inf
        elif mode == "train":
            im[self._train_mask, None] = array
            if add_inf is True:
                im[self._test_mask] = -np.inf
        else:
            im[:, None] = array
        if range is False:
            return im.reshape(self.points_per_dim, self.points_per_dim)
        else:
            vmin = np.nanmin(array)
            vmax = np.nanmax(array)
            return (
                im.reshape(self.points_per_dim, self.points_per_dim),
                vmin,
                vmax,
            )

    def plot_predictions(self, predictions):
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))

        test_im, tvmin, tvmax = self._make_im(self.test_responses, range=True)
        pred_im, pvmin, pvmax = self._make_im(predictions, range=True)
        vmin = np.min([tvmin, pvmin])
        vmax = np.max([tvmax, pvmax])

        self._label_ax(axes[0], "Expected Surface")
        im0 = axes[0].imshow(
            test_im,
            vmin=vmin,
            vmax=vmax,
            cmap="viridis",
        )
        fig.colorbar(im0, ax=axes[0])

        self._label_ax(axes[1], "Predicted Surface")
        im1 = axes[1].imshow(
            pred_im,
            vmin=vmin,
            vmax=vmax,
            cmap="viridis",
        )
        fig.colorbar(im1, ax=axes[1])

        plt.tight_layout()
        plt.show()

    def _get_images(self, predictions, confidence_intervals):
        resl_im, resl_min, resl_max = self._make_im(
            self.test_responses - predictions, range=True
        )
        conf_im, _, conf_mag = self._make_im(confidence_intervals, range=True)
        covr_im, covr_min, covr_max = self._make_im(
            np.abs(self.test_responses - predictions) - confidence_intervals,
            range=True,
        )

        resl_mag = np.max([np.abs(resl_min), np.abs(resl_max)])
        covr_mag = np.max([np.abs(covr_min), np.abs(covr_max)])
        return resl_im, conf_im, covr_im, resl_mag, conf_mag, covr_mag

    def plot_errors(self, *args):
        if len(args) % 3 != 0:
            raise ValueError("Only invocable on prediction/CI pairs!")

        row_count = int(len(args) / 3)

        fig, axes = plt.subplots(row_count, 3, figsize=(13, 4 * row_count))
        if row_count == 1:
            name, predictions, confidence_intervals = args
            (
                resl_im,
                conf_im,
                covr_im,
                resl_mag,
                conf_mag,
                covr_mag,
            ) = self._get_images(predictions, confidence_intervals)
            self.plot_error(
                fig,
                axes,
                name,
                resl_im,
                conf_im,
                covr_im,
                resl_mag,
                conf_mag,
                covr_mag,
            )
        else:
            resl_ims = []
            conf_ims = []
            covr_ims = []
            resl_mag = 0.0
            conf_mag = 0.0
            covr_mag = 0.0
            for i in range(row_count):
                predictions = args[3 * i + 1]
                confidence_intervals = args[3 * i + 2]
                (
                    resl_im,
                    conf_im,
                    covr_im,
                    resl_mag_,
                    conf_mag_,
                    covr_mag_,
                ) = self._get_images(predictions, confidence_intervals)
                resl_ims.append(resl_im)
                conf_ims.append(conf_im)
                covr_ims.append(covr_im)
                resl_mag = np.max([resl_mag, resl_mag_])
                conf_mag = np.max([conf_mag, conf_mag_])
                covr_mag = np.max([covr_mag, covr_mag_])
            for i in range(row_count):
                name = args[3 * i]
                self.plot_error(
                    fig,
                    axes[i, :],
                    name,
                    resl_ims[i],
                    conf_ims[i],
                    covr_ims[i],
                    resl_mag,
                    conf_mag,
                    covr_mag,
                )
        plt.tight_layout()
        plt.show()

    def _label_ax(self, ax, title):
        ax.set_title(title, fontsize=18)
        ax.set_xlabel("Axis 0", fontsize=14)
        ax.set_ylabel("Axis 1", fontsize=14)
        ax.set_xticks([])
        ax.set_yticks([])

    def plot_error(
        self,
        fig,
        axes,
        name,
        resl_im,
        conf_im,
        covr_im,
        resl_mag,
        conf_mag,
        covr_mag,
    ):
        self._label_ax(axes[0], f"{name} residual")
        im0 = axes[0].imshow(
            resl_im, vmin=-resl_mag, vmax=resl_mag, cmap="coolwarm"
        )
        fig.colorbar(im0, ax=axes[0])

        self._label_ax(axes[1], f"{name} CI Magnitude")
        im1 = axes[1].imshow(conf_im, vmin=0.0, vmax=conf_mag, cmap="inferno")
        fig.colorbar(im1, ax=axes[1])

        self._label_ax(axes[2], f"{name} |Residual| - CI")
        im2 = axes[2].imshow(
            covr_im, vmin=-covr_mag, vmax=covr_mag, cmap="coolwarm"
        )
        fig.colorbar(im2, ax=axes[2])


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
