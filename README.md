[![pipeline status](https://lc.llnl.gov/gitlab/muygps/MuyGPyS/badges/main/pipeline.svg)](https://lc.llnl.gov/gitlab/muygps/MuyGPyS/-/commits/main)

# Fast implementation of the MuyGPs Gaussian process hyperparameter estimation algorithm



## Installation


Installation instructions:
```
$ cd /path/to/this/repo
$ pip install -r requirements.txt
$ pip install -e .
```


## The Basics


MuyGPs is a GP estimation method that affords fast hyperparameter optimization by way of performing leave-one-out cross-validation.
MuyGPs achieves best-in-class speed and scalability by limiting inference to the information contained in k nearest neighborhoods for prediction locations for both hyperparameter optimization and tuning.


### Data format


`MuyGPyS` expects that each train or test observation corresponds to a row index in feature and response matrices.
In our examples we assume that data is bundled into `train` and `test` dicts possessing the string keys `"input"`, `"output"`, and `"lookup"`.
`train["input"]` should be a `(train_count, feature_count)`-shaped `numpy.ndarray` encoding the training observations.
`train["output"]` should be a `(train_count, response_count)`-shaped `numpy.ndarray` encoding the training targets, i.e. ground-truth 1-hot encoded class labels or regression targets.
`train["lookup"]` is a convenience structure used only for classification workflows, and should be a `(train_count,)`-shaped `numpy.ndarray` encoding the class ID.
`train["lookup"] = np.argmax(train["output"], axis=1)` unless we are using the 2-class uncertainty quantification workflow, in which case `train["lookup"] = 2 * np.argmax(train["output"], axis=1) - 1`.


### Constructing Nearest Neighbor Lookups


`MuyGPyS.neighbors.NN_Wrapper` is an api for tasking several KNN libraries with the construction of lookup indexes that empower fast inference.
The wrapper constructor expects the training features, the number of nearest neighbors, and a method string specifying which algorithm to use, as well as any additional kwargs used by the methods.
Supported implementations include [exact KNN using sklearn](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.NearestNeighbors.html) ("exact") and [approximate KNN using hnsw](https://github.com/nmslib/hnswlib) ("hnsw").

Construct exact and approximate  KNN data example with k = 10
```
In [1]: from MuyGPyS.neighors import NN_Wrapper 

In [2]: train, test = load_the_data()

In [3]: nn_count = 10

In [4]: exact_nbrs_lookup = NN_Wrapper(train["input"], nn_count, nn_method="exact", algorithm="ball_tree")

In [5]: approx_nbrs_lookup = NN_Wrapper(train["input"], nn_count, nn_method="hnsw", space="l2", M=16)
```

These lookup data structures are then usable to find nearest neighbors of queries in the training data.


### Sampling Batches of Data


MuyGPyS includes convenience functions for sampling batches of data from existing datasets.
These batches are returned in the form of row indices, both of the sampled data as well as their nearest neighbors.
Also included is the ability to sample "balanced" batches, where the data is partitioned by class and we attempt to sample as close to an equal number of items from each class as is possible. 

Sampling random and balanced (for classification) batches of 100 elements:
```
In [6]: from MuyGPyS.optimize.batch import sample_batch, get_balanced_batch

In [7]: batch_count, train_count = (100, train["input"].shape[0])

In [8]: batch_indices, batch_nn_indices = sample_batch(exact_nbrs_lookup, batch_count, train_count)

In [9]: balanced_indices, balanced_nn_indices = get_balanced_batch(exact_nbrs_lookup, train["lookup"], batch_count) # Classification only!
```

These `indices` and `nn_indices` arrays are the basic operating blocks of `MuyGPyS` linear algebraic inference.
The elements of `indices.shape == (batch_count,)` lists all of the row indices into `train`'s matrices corresponding to the sampled data.
The rows of `nn_indices.shape == (batch_count, nn_count)` list the row indices into `train`'s matrices corresponding to the nearest neighbors of the sampled data.
While the user need not use MuyGPyS sampling tools to construct these data, they will need to construct similar indices into their data in order to use MuyGPyS.


### Setting and Optimizing Hyperparameters


One initializes a MuyGPS object by indicating the kernel, as well as optionally specifying hyperparameters.

Creating a Matern kernel:
```
In [10]: from MuyGPyS.gp.muygps import MuyGPS 

In [11]: muygps = MuyGPS(kern="matern")
```

Hyperparameters are optionally set at initialization time or by using `set_params`.
```
In [12]: unset_params = muygps.set_params(length_scale=1.4, eps=1e-5, sigma_sq=[1.0])
```

Here `unset_params` is a list of kernel hyperparameters that have not been set, and is a convenient data structure for specifying optimization.
The MuyGPS object has default bounds for the optimization of its hyperparameters, but they can be overridden using `set_optim_bounds`:
```
In [13]: muygps.set_optim_bounds(nu=(1e-10, 1.5))
```

We supply a leave-one-out cross-validation objective functional for use with `scipy.optimize`.
```
In [14]: import numpy as np

In [15]: from scipy import optimize as opt

In [16]: from MuyGPyS.optimize.objective import loo_crossval, mse_fn

In [17]: bounds = muygps.optim_bounds(unset_params)

In [18]: x0 = np.array([np.random.uniform(low=b[0], high=b[1]) for b in bounds])

In [19]: optres = opt.minimize(loo_crossval, x0, args=(mse_fn, muygps, unset_params, batch_indices, batch_nn_indices, train["input"], train["output"]), method="L-BFGS-B", bounds=bounds)

In [20]: muygps.set_param_array(unset_params, optres.x)
```


### Inference


With set hyperparameters, we are able to use the `muygps` object to predict the response of test data.
Several workflows are supported.
See below a simple regression workflow, using the data structures built up in this example:
```
In [21]: test_indices = np.array([*range(test_count)])

In [22]: test_nn_indices = train_nbrs_lookup.get_nns(test["input"])

In [23]: predictions = muygps.regress(test_indices, test_nn_indices, test["input"], train["input"], train["output"])
```

More complex workflows are of course available.
See the `MuyGPyS.examples` high-level API functions for examples.


## API Examples


Listed below are several examples using the high-level APIs located in `MuyGPyS.examples.classify` and `MuyGPyS.examples.regress`.
Note that one need not go through these APIs to use `MuyGPyS`, but they condense many workflows into a single function call.

The example workflows below use Amanda's star-galaxy dataset.
One can of course replace the star-galaxy dataset with your data of choice, so long as it is contained within two Python dicts such as the `train` and `test` dicts as specified above.



## Classification


What follows is an example workflow performing two-class classification with uncertainty quantification.
Specific outputs uses a star-galaxy image dataset, where stars are labeled `[-1, +1]` and galaxies are labeled `[+1, -1]`.
Loading logic is encapsulated in the imaginary `load_stargal` function.
The workflow suffices for any conforming 2-class dataset.

What follows is example code surrounding the invocation of `MuyGPyS.examples.classify.do_classify`.
This function returns GP predictions `surrogate_predictions` and, if `uq_objectives is not None`, a list of index masks `masks`.  

Run star-gal with UQ example instructions:
```
Python 3.8.5 (default, Oct  5 2020, 15:42:46)
Type 'copyright', 'credits' or 'license' for more information
IPython 7.18.1 -- An enhanced Interactive Python. Type '?' for help.

In [1]: import numpy as np

In [2]: from MuyGPyS.examples.classify import do_classify, do_uq, example_lambdas

In [3]: train, test = load_stargal()

In [4]: surrogate_predictions, masks = do_classify(train, test, nn_count=50, embed_dim=50, opt_batch_size=500, uq_batch_size=2000, kern="matern", embed_method="pca", loss_method="log", hyper_dict={"eps": 0.015}, nn_kwargs={"nn_method": "hnsw", "space": "cosine"}, uq_objectives=example_lambdas, verbose=True)

parameters to be optimized: ['length_scale', 'nu']
bounds: [(1e-06, 40.0), (1e-06, 2.0)]
sampled x0: [35.18801977  0.89545343]
optimizer results: 
      fun: 72.1327677182081
 hess_inv: <2x2 LbfgsInvHessProduct with dtype=float64>
      jac: array([-3.97903934e-05,  1.05160324e-04])
  message: b'CONVERGENCE: REL_REDUCTION_OF_F_<=_FACTR*EPSMCH'
     nfev: 117
      nit: 17
     njev: 39
   status: 0
  success: True
        x: array([3.05693139, 0.66811425])
lkgp params : {'length_scale': 3.0569313854675695, 'nu': 0.6681142468721786}
cutoffs : [9.38, 19.6, 19.6, 19.6, 0.5]
timing : {'embed': 2.0149163901805878e-06, 'nn': 0.16120136599056423, 'batch': 0.08035180903971195, 'hyperopt': 13.377497965935618, 'pred': 2.7201196271926165, 'pred_full': ({'nn': 0.04120665090158582, 'agree': 0.0029976689256727695, 'pred': 2.6743266419507563},), 'uq_batch': 0.10259524686262012, 'cutoff': 0.912091915961355}

In [5]: predicted_labels = (surrogate_predictions > 0.0).astype(int)

In [6]: accuracy, uq = do_uq(predicted_labels, test, masks)

In [7]: print(f"Total accuracy : {accuracy}")
Out[7]: Total accuracy : 0.9762

In [8]: print(f"mask uq : \n{uq}")
Out[8]: mask uq : 
	[[8.21000000e+02 8.53836784e-01 9.87144569e-01]
 	[8.59000000e+02 8.55646100e-01 9.87528717e-01]
 	[1.03500000e+03 8.66666667e-01 9.88845510e-01]
 	[1.03500000e+03 8.66666667e-01 9.88845510e-01]
 	[5.80000000e+01 6.72413793e-01 9.77972239e-01]]
```

The kwarg `hyper_dict` expects a dictionary of fixed kernel hyperparameters that are not to be optimized.
If all kwargs are fixed - e.g. `eps`, `length_scale` and `nu` in the case of the `matern` kernel - then no optimization will occur.
If `hyper_dict=None` (default behavior), then we will optimize over all of the kernel hyperparameters.

The kwarge `nn_kwargs` expects a dictionary of kwargs for KNN library initialization, as well as the additional `nn_method` key.

`uq_objectives` expects a list of functions of `alpha`, `beta`, `correct_count`, and `incorrect_count`, where `alpha` and `beta` are the number of type I and type II errors, respectively.
`MuyGPyS.examples.classify.example_lambdas` lists some options, but you can supply your own.

If uncertainty quantification is not desired, or the classifcation problem in question involves more than two classes, set `uq_objectives=None`.
This is the default behavior.
In this case, there will be only one return value, i.e. `do_classify` will return `surrogate_predictions` but no `masks`.
Furthermore, whereas in the two class case with UQ the shape of `surrogate_predictions` is `(test_count,)`, where there is no UQ `surrogate_predictions` will contain a separate prediction for each class and will be of shape `(test_count, class_count)`. 


Run star-gal without UQ example instructions:
```
Python 3.8.5 (default, Oct  5 2020, 15:42:46)
Type 'copyright', 'credits' or 'license' for more information
IPython 7.18.1 -- An enhanced Interactive Python. Type '?' for help.

In [1]: import numpy as np

In [2]: from MuyGPyS.examples.classify import do_classify

In [3]: train, test = load_stargal()

In [4]: surrogate_predictions = do_classify(train, test, nn_count=50, embed_dim=50, opt_batch_size=500, kern="matern", embed_method="pca", loss_method="log", hyper_dict={"eps": 0.015}, nn_kwargs={"nn_method": "hnsw", "space": "cosine"}, verbose=True)

parameters to be optimized: ['length_scale', 'nu']
bounds: [(1e-06, 40.0), (1e-06, 2.0)]
sampled x0: [35.18801977  0.89545343]
optimizer results: 
      fun: 72.1327677182081
 hess_inv: <2x2 LbfgsInvHessProduct with dtype=float64>
      jac: array([-3.97903934e-05,  1.05160324e-04])
  message: b'CONVERGENCE: REL_REDUCTION_OF_F_<=_FACTR*EPSMCH'
     nfev: 117
      nit: 17
     njev: 39
   status: 0
  success: True
        x: array([3.05693139, 0.66811425])
lkgp params : {'length_scale': 3.0569313854675695, 'nu': 0.6681142468721786}
cutoffs : [9.38, 19.6, 19.6, 19.6, 0.5]
timing : {'embed': 2.0149163901805878e-06, 'nn': 0.16120136599056423, 'batch': 0.08035180903971195, 'hyperopt': 13.377497965935618, 'pred': 2.7201196271926165, 'pred_full': ({'nn': 0.04120665090158582, 'agree': 0.0029976689256727695, 'pred': 2.6743266419507563},), 'uq_batch': 0.10259524686262012, 'cutoff': 0.912091915961355}

In [5]: predicted_labels = np.argmax(surrogate_predictions, axis=1)

In [6]: print(f"Total accuracy : {np.sum(predicted_labels == test["lookup"])}")
Out[6]: Total accuracy : 0.9762
```

## Regression


We can use a similar API to perform regression.
The following example uses the star-galaxy data as above, but any conforming data might be used.
If one wants to predict on a univariate response, one must ensure the data is stored as a matrix rather than as a vector, i.e. that `train['output'].shape = (train_count, 1)`.
The regression API adds a `sigma_sq` scale parameter for the variance.
One can set `sigma_sq` using the `hyper_dict` kwarg like other hyperparameters.
The API expects that `sigma_sq` is a `numpy.ndarray` with a value associated with each dimension of the response, i.e. that `train['output'].shape[1] == len(sigma_sq)`.

Run star-gal with no variance
```
Python 3.8.5 (default, Oct  5 2020, 15:42:46)
Type 'copyright', 'credits' or 'license' for more information
IPython 7.18.1 -- An enhanced Interactive Python. Type '?' for help.

In [1]: import numpy as np

In [2]: from MuyGPyS.examples.regress import do_regress

In [3]: train, test = load_stargal()

In [4]: predictions = do_regress(train, test, nn_count=50, embed_dim=50, batch_size=500, kern="matern", embed_method="pca", loss_method="mse", hyper_dict={"eps": 0.015}, variance_mode=diagonal, nn_kwargs={"nn_method": "hnsw", "space": "cosine"}, verbose=True)

optimization parameters: ['length_scale', 'nu']
bounds: [(1e-06, 40.0), (1e-06, 2.0)]
sampled x0: [9.1479526  1.13113405]
optimizer results: 
      fun: 0.06288895645420908
 hess_inv: <2x2 LbfgsInvHessProduct with dtype=float64>
      jac: array([ 1.84574579e-07, -9.53542795e-06])
  message: b'CONVERGENCE: NORM_OF_PROJECTED_GRADIENT_<=_PGTOL'
     nfev: 48
      nit: 7
     njev: 16
   status: 0
  success: True
        x: array([6.77309969, 0.54720995])
lkgp params : {'length_scale': 6.77309968633532, 'nu': 0.5472099500708325}
timing : {'embed': 1.2968666851520538e-06, 'nn': 0.16625570296309888, 'batch': 3.210734575986862e-07, 'hyperopt': 3.7612421449739486, 'pred': 5.0821877010166645, 'pred_full': {'nn': 0.03705085511319339, 'agree': 8.50064679980278e-07, 'pred': 5.044568303972483}}


In [5]: from MuyGPyS.optimize.objective import mse_fn

In [6]: print(f"MSE : {mse_fn(predictions, test["output"])}")
Out[6]: MSE: 0.09194243606326429

In [7]: print(f"Accuracy : {np.mean(np.argmax(predictions, axis=1) == np.argmax(test["output"]))}")
Out[7]: Accuracy: 0.9744

In [8]: print(f"Variance: {variance}")
Out [8]: Variance: [0.0220988  0.01721446 0.02487733 ... 0.03346663 0.17433852 0.03273306]
```

If one requires the (individual, independent) posterior variances for each of the predictions, one can pass `variance_mode="diagonal"`.
This mode assumes that each output dimension uses the same model, and so will output a vector `variance` with a scalar posterior variance associated with each test point.
The API also returns `sigma_sq`, which reports a multiplicative scaling parameter on the variance of each dimension.
Obtaining the tuned posterior variance implies multiplying the returned variance by the scaling parameter along each dimension.


Run star-gal with diagonal variance
```
Python 3.8.5 (default, Oct  5 2020, 15:42:46)
Type 'copyright', 'credits' or 'license' for more information
IPython 7.18.1 -- An enhanced Interactive Python. Type '?' for help.

In [1]: import numpy as np

In [2]: from MuyGPyS.examples.regress import do_regress

In [3]: train, test = load_stargal()

In [4]: predictions, variance, sigma_sq = do_regress(train, test, nn_count=50, embed_dim=50, batch_size=500, kern="matern", embed_method="pca", loss_method="mse", hyper_dict={"eps": 0.015}, variance_mode=diagonal, nn_kwargs={"nn_method": "hnsw", "space": "cosine"}, verbose=True)

optimization parameters: ['length_scale', 'nu']
bounds: [(1e-06, 40.0), (1e-06, 2.0)]
sampled x0: [9.1479526  1.13113405]
optimizer results: 
      fun: 0.06288895645420908
 hess_inv: <2x2 LbfgsInvHessProduct with dtype=float64>
      jac: array([ 1.84574579e-07, -9.53542795e-06])
  message: b'CONVERGENCE: NORM_OF_PROJECTED_GRADIENT_<=_PGTOL'
     nfev: 48
      nit: 7
     njev: 16
   status: 0
  success: True
        x: array([6.77309969, 0.54720995])
lkgp params : {'length_scale': 6.77309968633532, 'nu': 0.5472099500708325}
timing : {'embed': 1.2968666851520538e-06, 'nn': 0.16625570296309888, 'batch': 3.210734575986862e-07, 'hyperopt': 3.7612421449739486, 'pred': 5.0821877010166645, 'pred_full': {'nn': 0.03705085511319339, 'agree': 8.50064679980278e-07, 'pred': 5.044568303972483}}

In [5]: from MuyGPyS.optimize.objective import mse_fn

In [6]: print(f"MSE : {mse_fn(predictions, test["output"])}")
Out[6]: MSE: 0.09194243606326429

In [7]: print(f"Accuracy : {np.mean(np.argmax(predictions, axis=1) == np.argmax(test["output"], axis=1))}")
Out[7]: Accuracy: 0.9744

In [8]: print(f"Variance: {variance * sigma_sq[0]}")
Out [8]: Variance: [0.0220988  0.01721446 0.02487733 ... 0.03346663 0.17433852 0.03273306]
```

This is presently the only form of posterior variance collection that is supported.
Computing the independent diagonal posterior variances between the dimensions of multivariate output with different models is not currently supported.
Computing the full posterior covariance between the dimensions of multivariate output is not currently supported.
Computing the full posterior covariance between all inputs is not and will not be supported for scalability reasons. 


## Optional workflow modifications for experiment chassis design


What follows are some quality-of-life modifications to workflows involving repeated invocations of the regression or classification APIs on the same dataset.


### Preprocessing data embedding for many trials


If one wants to experiment with many trials with the same embedding dimension with a deterministic(-ish) embedding method like PCA, on cane use `MuyGPyS.embed.embed_all` to embed the data either in-place or into a pair of fresh dicts, depending on your experiment needs.

In-place embedding:
```
In [1] from MuyGPyS.embed import embed_all 

In [2] embed_all(train, test, embed_dim=40, embed_method="pca", in_place=True)
```

Copy embedding:
```
In [3] embedded_train, embedded_test = embed_all(train, test, embed_dim=40, embed_method="pca", in_place=False)
```

In either case, you should pass the appropriate `train` and `test` dicts to `do_classify` or `do_regress` along with the kwarg `embed_method=None` so that the API knows not to try to embed the data again.


### Sampling smaller datasets


Similarly, one might want to run trials using a smaller number of samples than the full dataset, perhaps as part of an exploration experiment where you invoke the API many times.
In addition to possibly invoking `MuyGPyS.embed.embed_all` as above, can use `MuyGPyS.data.utils.subsample` or `MuyGPyS.data.utils.balanced_subsample`.
The latter utility is reserved for classification, and tries to collect an equal number of samples of each class.

Subsample example:
```
In [1] from MuyGPyS.data.utils import subsample

In [2] sub_train = subsample(train, 1000)

In [3] sub_test = subsample(test, 1000)
```

Balanced subsample example:
```
In [1] from MuyGPyS.data.utils import subsample

In [2] sub_train = balanced_subsample(train, 1000)

In [3] sub_test = balanced_subsample(test, 1000)
```


### Specifying hyperparameter bounds for optimization


When training hyperparameters, one might want to apply prior knowledge to constrain the range of reasonable values that the optimizer should consider.
`MuyGPyS` has built-in bounds that can be overridden with the `optim_bounds` kwarg, which expects a dict mapping hyperparameter name strings to 2-tuples specifying the lower and upper bounds to be considered for optimization, respectively.

`optim_bounds` example:
```
In [1]: predictions = do_regress(train, test, nn_count=50, embed_dim=50, batch_size=500, kern="matern", embed_method="pca", loss_method="mse", hyper_dict={"length_scale": 1.5, "eps": 0.015}, nn_kwargs={"nn_method": "hnsw", "space": "cosine"}, optim_bounds={"nu": (1e-5, 1.0)}, variance_mode=None, verbose=True)

optimization parameters: ['nu']
bounds: [(1e-05, 1.0)]
sampled x0: [0.1479526]
optimizer results: 
      fun: 0.06288895645420908
 hess_inv: <2x2 LbfgsInvHessProduct with dtype=float64>
      jac: array([ 1.84574579e-07, -9.53542795e-06])
  message: b'CONVERGENCE: NORM_OF_PROJECTED_GRADIENT_<=_PGTOL'
     nfev: 48
      nit: 7
     njev: 16
   status: 0
  success: True
        x: array([0.54720995])
lkgp params : {'nu': 0.5472099500708325}
timing : {'embed': 1.2968666851520538e-06, 'nn': 0.16625570296309888, 'batch': 3.210734575986862e-07, 'hyperopt': 3.7612421449739486, 'pred': 5.0821877010166645, 'pred_full': {'nn': 0.03705085511319339, 'agree': 8.50064679980278e-07, 'pred': 5.044568303972483}}
```

One should specify bounds using `optim_bounds` only for hyperparameters that are not specified in `hyper_dict`, as those hyperparameters will be fixed.


# About

## Authors

* Benjamin W. Priest (priest2 at llnl dot gov)
* Amanada Muyskens (muyskens1 at llnl dot gov)

## Papers

MuyGPyS has been used the in the following papers (newest first):

1. [Star-Galaxy Separation via Gaussian Processes with Model Reduction](https://arxiv.org/abs/2010.06094)

## License

MuyGPyS is distributed under the terms of both the MIT license and the Apache License (Version 2.0).
Users may choose either license, at their discretion.

All new contributions must be made under both the MIT and Apache-2.0 licenses.

See [LICENSE-MIT](LICENSE-MIT), [LICENSE-APACHE](LICENSE-APACHE), [NOTICE](NOTICE), and [COPYRIGHT](COPYRIGHT) for details.

SPDX-License-Identifier: (Apache-2.0 or MIT)

## Release

LLNL-CODE-XXXXXX