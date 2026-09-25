import numpy as np
import matplotlib.pyplot as plt

#MuyGPyS imports
from MuyGPyS.gp.deformation import F2, Isotropy, l2
from MuyGPyS.gp.hyperparameter import AnalyticScale, Parameter
from MuyGPyS.gp.kernels import RBF as RBF
from MuyGPyS.gp.noise import HomoscedasticNoise
from MuyGPyS.optimize.loss import *
from muygps_wrapper_class import MuyGPSWrapper


def main():
    
    
    X = np.linspace(start=0, stop=10, num=1000).reshape(-1, 1)
    y = np.squeeze(X * np.sin(X))
    
    rng = np.random.RandomState(1)
    training_indices = rng.choice(np.arange(y.size), size=10, replace=False)
    X_train, y_train = X[training_indices], y[training_indices]

    train_count = len(X_train)
    nn_count = np.min([train_count-1, 30])

    k_kwargs = {
    "kernel": RBF(
        deformation=Isotropy(
            metric=F2,
            length_scale=Parameter(1.0, (1e-2, 1e2))
        )
    ),
    "noise": HomoscedasticNoise(1e-5),
    "scale": AnalyticScale(),
    }
    muygps = MuyGPSWrapper(**k_kwargs)
    muygps_optimized, nbrs_lookup  = muygps.fit_regressor(X_train, y_train, nn_count=nn_count, loss_fn=lool_fn)

    test_count = len(X)
    predictions = np.zeros((test_count))
    variances = np.zeros((test_count))
    for i in range(test_count): #test_count
        predictions_i, variances_i = muygps_optimized.predict_one(test_feature=X[i], train_features=X_train, train_targets=y_train, train_nbrs_lookup=nbrs_lookup)
        predictions[i] = predictions_i
        variances[i] = variances_i
    confidence_intervals = np.sqrt(variances) * 1.96

    plt.figure()
    plt.plot(X, y, label=r"$f(x) = x \sin(x)$", linestyle="dotted")
    plt.scatter(X_train, y_train, label="Observations")
    plt.plot(X, predictions, label="Mean prediction")
    plt.fill_between(
        X.ravel(),
        predictions - confidence_intervals,
        predictions + confidence_intervals,
        alpha=0.5,
        label=r"95% confidence interval",
    )
    plt.legend()
    plt.xlabel("$x$")
    plt.ylabel("$f(x)$")
    _ = plt.title("MuyGPys GP Fit")
    plt.savefig("muygps_gp_fit_predict_one.png")
    plt.show()

    return 0


if __name__=='__main__':
    main()