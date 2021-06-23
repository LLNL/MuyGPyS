[![pipeline status](https://lc.llnl.gov/gitlab/muygps/MuyGPyS/badges/main/pipeline.svg)](https://lc.llnl.gov/gitlab/muygps/MuyGPyS/-/commits/main)

# Fast implementation of the MuyGPs Gaussian process hyperparameter estimation algorithm


MuyGPs is a GP estimation method that affords fast hyperparameter optimization by way of performing leave-one-out cross-validation.
MuyGPs achieves best-in-class speed and scalability by limiting inference to the information contained in k nearest neighborhoods for prediction locations for both hyperparameter optimization and tuning.
This feature affords the optimization of hyperparameters by way of leave-one-out cross-validation, as opposed to the more expensive loglikelihood evaluations requires by similar sparse methods. 


## Installation


Installation instructions:
```
$ cd /path/to/this/repo
$ pip install -r requirements.txt
$ pip install -e .
```


## The Basics


### Data format


`MuyGPyS` expects that each train or test observation corresponds to a row index in feature and response matrices.
In our examples we assume that data is bundled into `train` and `test` dicts possessing the string keys `"input"` and `"output"`.
`train["input"]` should be a `(train_count, feature_count)`-shaped `numpy.ndarray` encoding the training observations.
`train["output"]` should be a `(train_count, response_count)`-shaped `numpy.ndarray` encoding the training targets, i.e. ground truth regression targets or 1-hot encoded class labels.


### Constructing Nearest Neighbor Lookups


`MuyGPyS.neighbors.NN_Wrapper` is an api for tasking several KNN libraries with the construction of lookup indexes that empower fast training and inference.
The wrapper constructor expects the training features, the number of nearest neighbors, and a method string specifying which algorithm to use, as well as any additional kwargs used by the methods.
Currently supported implementations include [exact KNN using sklearn](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.NearestNeighbors.html) ("exact") and [approximate KNN using hnsw](https://github.com/nmslib/hnswlib) ("hnsw").

Construct exact and approximate  KNN data example with k = 10
```
>>> from MuyGPyS.neighors import NN_Wrapper 
>>> train, test = load_the_data()
>>> nn_count = 10
>>> exact_nbrs_lookup = NN_Wrapper(train["input"], nn_count, nn_method="exact", algorithm="ball_tree")
>>> approx_nbrs_lookup = NN_Wrapper(train["input"], nn_count, nn_method="hnsw", space="l2", M=16)
```

These lookup data structures are then usable to find nearest neighbors of queries in the training data.


### Sampling Batches of Data


MuyGPyS includes convenience functions for sampling batches of data from existing datasets.
These batches are returned in the form of row indices, both of the sampled data as well as their nearest neighbors.
Also included is the ability to sample "balanced" batches, where the data is partitioned by class and we attempt to sample as close to an equal number of items from each class as is possible. 

Sampling random and balanced (for classification) batches of 100 elements:
```
>>> from MuyGPyS.optimize.batch import sample_batch, get_balanced_batch
>>> batch_count = 200
>>> train_count, _ = train["input"]
>>> batch_indices, batch_nn_indices = sample_batch(
...         exact_nbrs_lookup, batch_count, train_count
... )
>>> train_labels = np.argmax(train["output"], axis=1)
>>> balanced_indices, balanced_nn_indices = get_balanced_batch(
...         exact_nbrs_lookup, train_labels, batch_count
... ) # Classification only!
```

These `indices` and `nn_indices` arrays are the basic operating blocks of `MuyGPyS` linear algebraic inference.
The elements of `indices.shape == (batch_count,)` lists all of the row indices into `train`'s matrices corresponding to the sampled data.
The rows of `nn_indices.shape == (batch_count, nn_count)` list the row indices into `train`'s matrices corresponding to the nearest neighbors of the sampled data.
While the user need not use MuyGPyS sampling tools to construct these data, they will need to construct similar indices into their data in order to use MuyGPyS.


### Setting and Optimizing Hyperparameters


One initializes a MuyGPS object by indicating the kernel, as well as optionally specifying hyperparameters.

Creating a Matern kernel:
```
>>> from MuyGPyS.gp.muygps import MuyGPS
>>> k_kwargs = {
...         "kern": "rbf",
...         "metric": "F2",
...         "eps": {"val": 1e-5},
...         "nu": {"val": 0.38, "bounds": (0.1, 2.5)},
...         "length_scale": {"val": 7.2},
... }
>>> muygps = MuyGPS(**k_kwarg)
```

Hyperparameters can be initialized or reset using dictionary arguments containing the optional `"val"` and `"bounds"` keys.
`"val"` sets the hyperparameter to the given value, and `"bounds"` determines the upper and lower bounds to be used for optimization.
If `"bounds"` is set, `"val"` can also take the arguments `"sample"` and `"log_sample"` to generate a uniform or log uniform sample, respectively.
If `"bounds"` is set to `"fixed"`, the hyperparameter will remain fixed during any optimization.
This is the default behavior for all hyperparameters if `"bounds"` is unset by the user.

One sets yyperparameters such as `eps`, `sigma_sq`, as well as kernel-specific hyperparameters, e.g. `nu` and  `length_scale` for the Matern kernel, at initialization as above.
Alternately, one can reset parameters after initialization by invoking the `MuyGPyS.gp.muygps.MuyGPS.set_eps`,`MuyGPyS.gp.muygps.MuyGPS.set_sigma_sq`, or `MuyGPyS.gp.kernel.KernelFn.set_params` member functions.
```
>>> muygps.eps(val= 1.4, bounds=(1e-2, 1e-2))
>>> muygps.set_params([{"val": 1.0, "bounds": "fixed}])
>>> muygps.kernel.set_params(
...         length_scale={"val": 1.4, "bounds": (1e-2, 1e-2)}
... )
```

MuyGPyS depends upon linear operations on specially-constructed tensors in order to efficiently estimate GP realizations.
Constructing these tensors depends upon the nearest neighbor index matrices that we described above.
We can construct a distance tensor coalescing all of the square pairwise distance matrices of the nearest neighbors of a batch of points.
This snippet constructs a Euclidean distance tensor.
```
>>> from MuyGPyS.gp.distance import pairwise_distances
>>> pairwise_dists = pairwise_distances(
...         train['input'], batch_nn_indices, metric="l2"
... )
```

We can similarly construct a matrix coalescing all of the distance vectors between the same batch of points and their nearest neighbors.
```
>>> from MuyGPyS.gp.distance import crosswise_distances
>>> crosswise_dists = crosswise_distances(
...         train['input'],
...         train['input'],
...         batch_indices,
...         batch_nn_indices,
...         metric='l2',
... )
```

We can easily realize kernel tensors using a `MuyGPS` object's kernel functor:
```
>>> K = muygps.kernel(pairwise_dists)
>>> Kcross = muygps.kernel(crosswise_dists)
```

We supply a convenient leave-one-out cross-validation utility that internally realizes kernel tensors in this manner.
```
>>> from MuyGPyS.optimize.chassis.scipy_optimize_from_tensors
>>> scipy_optimize_from_tensors(
...         muygps,
...         batch_indices,
...         batch_nn_indices,
...         crosswise_dists,
...         pairwise_dists,
...         train['output'],
...         loss_method="mse",
...         verbose=True,
... )
parameters to be optimized: ['nu']
bounds: [[0.1 1. ]]
sampled x0: [0.8858425]
optimizer results:
      fun: 0.4797763813693626
 hess_inv: <1x1 LbfgsInvHessProduct with dtype=float64>
      jac: array([-3.06976666e-06])
  message: b'CONVERGENCE: NORM_OF_PROJECTED_GRADIENT_<=_PGTOL'
     nfev: 16
      nit: 5
     njev: 8
   status: 0
  success: True
        x: array([0.39963594])
```

If you do not need to keep the distance tensors around for reference, you can use a related function:
```
>>> from MuyGPyS.optimize.chassis.scipy_optimize_from_indices
>>> scipy_optimize_from_indices(
...         muygps,
...         batch_indices,
...         batch_nn_indices,
...         train['input'],
...	    test['input'],
...	    train['output'],
...         loss_method="mse",
...         verbose=False,
... )
```


### One-line model creation


The library also provides one-line APIs for creating MuyGPs models intended for regression and classification.
The functions are `MuyGPyS.examples.regress.make_regressor` and `MuyGPyS.examples.classify.make_classifier`, respectively.
These functions provide convenient mechanisms for specifying and optimizing models if you have no need to later reference their intermediate data structures (such as the training batches or their distance tensors).
They return only the trained `MuyGPyS.gp.muygps.MuyGPS` model and the `MuyGPyS.neighbors.NN_Wrapper` neighbors lookup data structure.

An example regressor.
In order to automatically train `sigma_sq`, set `k_kwargs["sigma_sq"] = "learn"`. 
```
>>> from MuyGPyS.examples.regress import make_regressor
>>> train, test = load_regression_dataset()  # hypothetical data load
>>> nn_kwargs = {"nn_method": "exact", "algorithm": "ball_tree"}
>>> k_kwargs = {
...         "kern": "rbf",
...         "metric": "F2",
...         "eps": {"val": 1e-5},
...         "nu": {"val": 0.38, "bounds": (0.1, 2.5)},
...         "length_scale": {"val": 7.2},
...         "sigma_sq": "learn",
... }
>>> muygps, nbrs_lookup = make_regressor(
...         train["input"],
...         train["output"],
...         nn_count=40,
...         batch_size=500,
...         loss_method="mse",
...         k_kwargs=k_kwargs,
...         nn_kwargs=nn_kwargs,
...         verbose=False,
... )    
```

An example surrogate classifier.
```
>>> from MuyGPyS.examples.classify import make_classifier
>>> train, test = load_classification_dataset()  # hypothetical data load
>>> nn_kwargs = {"nn_method": "exact", "algorithm": "ball_tree"}
>>> k_kwargs = {
...         "kern": "rbf",
...         "metric": "F2",
...         "eps": {"val": 1e-5},
...         "nu": {"val": 0.38, "bounds": (0.1, 2.5)},
...         "length_scale": {"val": 7.2},
... }
>>> muygps, nbrs_lookup = make_classifier(
...         train["input"],
...         train["output"],
...         nn_count=40,
...         batch_size=500,
...         loss_method="log",
...         k_kwargs=k_kwargs,
...         nn_kwargs=nn_kwargs,
...         verbose=False,
... )    
```


### Inference


With set hyperparameters, we are able to use the `muygps` object to predict the response of test data.
Several workflows are supported.
See below a simple regression workflow, using the data structures built up in this example:
```
>>> indices = np.arange(test_count)
>>> nn_indices = train_nbrs_lookup.get_nns(test["input"])
>>> pairwise_dists = pairwise_distances(
...         train['input'], batch_nn_indices, metric="l2"
... )
>>> crosswise_dists = crosswise_distances(
...         test['input'],
...         train['input'],
...         indices,
...         nn_indices,
...         metric='l2',
... )
>>> K = muygps.kernel(pairwise_dists)
>>> Kcross = muygps.kernel(crosswise_dists)
>>> predictions = muygps.regress(K, Kcross, train['output'][nn_indices, :])
```

Again if you do not want to reuse your tensors, you can run the more compact:
```
>>> indices = np.arange(test_count)
>>> nn_indices = train_nbrs_lookup.get_nns(test["input"])
>>> muygps.regress_from_indices(
...         indices,
...	    nn_indices,
...	    test['input'],
...	    train['input'],
...	    train['output'],
... )
```

More complex workflows are of course available.
See the `MuyGPyS.examples` high-level API functions for examples.


## API Examples


Listed below are several examples using the high-level APIs located in `MuyGPyS.examples.classify` and `MuyGPyS.examples.regress`.
Note that one need not go through these APIs to use `MuyGPyS`, but they condense many workflows into a single function call.
In all of these examples, note that if all of the hyperparameters are fixed in `k_kwargs` (i.e. you supply no optimization bounds), the API will perform no optimization and will instead simply predict on the data problem using the provided kernel.


## Regression


The following example performs GP regression on the [Heaton spatial statistics case study dataset](https://github.com/finnlindgren/heatoncomparison).
In the example, `load_heaton` is a unspecified function that reads in the dataset in the specified dict format.
In practice, a user can use any conforming dataset.
If one wants to predict on a univariate response as in this example, one must ensure the data is stored as a matrix rather than as a vector, i.e. that `train['output'].shape = (train_count, 1)`.
The regression API adds a `sigma_sq` scale parameter for the variance.
One can set `sigma_sq` using the `hyper_dict` kwarg like other hyperparameters.
The API expects that `sigma_sq` is a `numpy.ndarray` with a value associated with each dimension of the response, i.e. that `train['output'].shape[1] == len(sigma_sq)`.

Regress on Heaton data with no variance
```
>>> import numpy as np
>>> from MuyGPyS.examples.regress import do_regress
>>> from MuyGPyS.optimize.objective import mse_fn
>>> train, test = load_heaton()
>>> nn_kwargs = {"nn_method": "exact", "algorithm": "ball_tree"}
>>> k_kwargs = {
...         "kern": "rbf",
...         "metric": "F2",
...         "eps": {"val": 1e-5},
...         "nu": {"val": 0.38, "bounds": (0.1, 2.5)},
...         "length_scale": {"val": 7.2},
... }
>>> muygps, nbrs_lookup, predictions = do_regress(
...         test['input'],
...         train['input'],
...         train['output'],
...         nn_count=30,
...         batch_size=200,
...         loss_method="mse",
...         variance_mode=None,
...         k_kwargs=k_kwargs,
...         nn_kwargs=nn_kwargs,
...         verbose=True,
... )
parameters to be optimized: ['nu']
bounds: [[0.1 1. ]]
sampled x0: [0.8858425]
optimizer results:
      fun: 0.4797763813693626
 hess_inv: <1x1 LbfgsInvHessProduct with dtype=float64>
      jac: array([-3.06976666e-06])
  message: b'CONVERGENCE: NORM_OF_PROJECTED_GRADIENT_<=_PGTOL'
     nfev: 16
      nit: 5
     njev: 8
   status: 0
  success: True
        x: array([0.39963594])
NN lookup creation time: 0.04974837500000007s
batch sampling time: 0.017116840000000022s
tensor creation time: 0.14439213699999998s
hyper opt time: 2.742181974s
sigma_sq opt time: 5.359999999399179e-07s
prediction time breakdown:
        nn time:0.1069446140000001s
        agree time:1.9999999967268423e-07s
        pred time:10.363597161000001s
finds hyperparameters:
        nu : 0.8858424985399979
>>> print(f"mse : {mse_fn(predictions, test["output"])}")
obtains mse: 2.345136495565052
```

If one requires the (individual, independent) posterior variances for each of the predictions, one can pass `variance_mode="diagonal"`.
This mode assumes that each output dimension uses the same model, and so will output an additional vector `variance` with a scalar posterior variance associated with each test point.
The API also returns a (possibly trained) `MuyGPyS.gp.MuyGPS` instance, whose `sigma_sq` member reports an array of multiplicative scaling parameters associated with the variance of each dimension.
In order to tune `sigma-sq` using the `do_regress` API, pass `k_kwargs["sigma_sq"] = "learn"`.
Obtaining the tuned posterior variance implies multiplying the returned variance by the scaling parameter along each dimension.


Regress on Heaton data while estimating diagonal variance
```
>>> import numpy as np
>>> from MuyGPyS.examples.regress import do_regress
>>> from MuyGPyS.optimize.objective import mse_fn
>>> train, test = load_heaton()
>>> nn_kwargs = {"nn_method": "exact", "algorithm": "ball_tree"}
>>> k_kwargs = {
...         "kern": "rbf",
...         "metric": "F2",
...         "eps": {"val": 1e-5},
...         "nu": {"val": 0.38, "bounds": (0.1, 2.5)},
...         "length_scale": {"val": 7.2},
...	    "sigma_sq": "learn",
... }
>>> muygps, nbrs_lookup, predictions, variance = do_regress(
...         test['input'],
...         train['input'],
...         train['output'],
...         nn_count=30,
...         batch_size=200,
...         loss_method="mse",
...         variance_mode="diagonal",
...         k_kwargs=k_kwargs,
...         nn_kwargs=nn_kwargs,
...         verbose=False,
... )
>>> print(f"mse : {mse_fn(predictions, test["output"])}")
obtains mse: 2.345136495565052
>>> print(f"diagonal posterior variance: {variance * muygps.sigma_sq()}")
diagonal posterior variance: [0.52199482 0.45934382 0.81381388 ... 0.64982631 0.45958342 0.68602048]
```

This is presently the only form of posterior variance collection that is supported.
Computing the independent diagonal posterior variances between the dimensions of multivariate output with different models is not currently supported, but is planned for a future release.
Computing the full posterior covariance between the dimensions of multivariate output is not currently supported, but is planned for a future release.
Computing the full posterior covariance between all inputs is not and will not be supported for scalability reasons. 



## Classification


What follows is an example workflow performing two-class classification with uncertainty quantification.
Specific outputs uses a star-galaxy image dataset, where stars are labeled `[-1, +1]` and galaxies are labeled `[+1, -1]`.
Loading logic is encapsulated in the imaginary `load_stargal` function.
The workflow suffices for any conforming 2-class dataset.

What follows is example code surrounding the invocation of `MuyGPyS.examples.classify.do_classify_uq`.
This function returns GP predictions `surrogate_predictions` and a list of index masks `masks`.  

Run star-gal with UQ example instructions:
```
>>> import numpy as np
>>> from MuyGPyS.examples.classify import do_classify_uq, do_uq, example_lambdas
>>> from MuyGPyS.optimize.objective import mse_fn
>>> train, test = load_stargal()
>>> nn_kwargs = {"nn_method": "exact", "algorithm": "ball_tree"}
>>> k_kwargs = {
...         "kern": "rbf",
...         "metric": "F2",
...         "eps": {"val": 1e-5},
...         "nu": {"val": 0.38, "bounds": (0.1, 2.5)},
...         "length_scale": {"val": 7.2},
... }
>>> muygps, nbrs_lookup, surrogate_predictions, masks = do_classify_uq(
...         test['input'],
...         train['input'],
...         train['output'],
...         nn_count=30,
...         opt_batch_size=200,
...	    uq_batch_size=500,
...         loss_method="log",
...         variance_mode=None,
...	    uq_objectives=example_lambdas,
...         k_kwargs=k_kwargs,
...         nn_kwargs=nn_kwargs,
...         verbose=False,
... )
>>> accuracy, uq = do_uq(surrogate_predictions, test["output"], masks)
>>> print(f"obtained accuracy: {accuracy}")
obtained accuracy: 0.973...
>>> print(f"mask uq : \n{uq}")
mask uq : 
[[8.21000000e+02 8.53836784e-01 9.87144569e-01]
 [8.59000000e+02 8.55646100e-01 9.87528717e-01]
 [1.03500000e+03 8.66666667e-01 9.88845510e-01]
 [1.03500000e+03 8.66666667e-01 9.88845510e-01]
 [5.80000000e+01 6.72413793e-01 9.77972239e-01]]
```

`uq_objectives` expects a list of functions of `alpha`, `beta`, `correct_count`, and `incorrect_count`, where `alpha` and `beta` are the number of type I and type II errors, respectively.
`MuyGPyS.examples.classify.example_lambdas` lists some options, but you can supply your own.

If uncertainty quantification is not desired, or the classifcation problem in question involves more than two classes, instead use a workflow like that in `MuyGPyS.examples.classify.do_classify`.

Run MNIST without UQ example instructions:
```
>>> import numpy as np
>>> from MuyGPyS.examples.classify import do_classify
>>> from MuyGPyS.optimize.objective import mse_fn
>>> train, test = load_mnist()
>>> nn_kwargs = {"nn_method": "exact", "algorithm": "ball_tree"}
>>> k_kwargs = {
...         "kern": "rbf",
...         "metric": "F2",
...         "eps": {"val": 1e-5},
...         "nu": {"val": 0.38, "bounds": (0.1, 2.5)},
...         "length_scale": {"val": 7.2},
... }
>>> muygps, nbrs_lookup, surrogate_predictions = do_classify(
...         test['input'],
...         train['input'],
...         train['output'],
...         nn_count=30,
...         batch_size=200,
...         loss_method="log",
...         variance_mode=None,
...         k_kwargs=k_kwargs,
...         nn_kwargs=nn_kwargs,
...         verbose=False,
... )
>>> predicted_labels = np.argmax(surrogate_predictions, axis=1)
>>> true_labels = np.argmax(test['output'], axis=1)
>>> accuracy = np.mean(predicted_labels == true_labels)
>>> print(f"obtained accuracy: {accuracy}")
0.97634
```


## Optional workflow modifications for experiment chassis design


What follows are some quality-of-life modifications to workflows involving repeated invocations of the regression or classification APIs on the same dataset.



### Sampling smaller datasets


One might want to run trials using a smaller number of samples than the full dataset, perhaps as part of an exploration experiment where you invoke the API many times.
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


# About

## Authors

* Benjamin W. Priest (priest2 at llnl dot gov)
* Amanada Muyskens (muyskens1 at llnl dot gov)

## Citation

If you use MuyGPyS in a research paper, please reference our article:

```
@article{muygps2021,
  title={MuyGPs: Scalable Gaussian Process Hyperparameter Estimation Using Local Cross-Validation},
  author={Muyskens, Amanda and Priest, Benjamin W. and Goumiri, Im{\`e}ne and Schneider, Michael},
  journal={arXiv preprint arXiv:2104.14581},
  year={2021}
}

```

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
