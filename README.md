[![pipeline status](https://lc.llnl.gov/gitlab/muygps/MuyGPyS/badges/main/pipeline.svg)](https://lc.llnl.gov/gitlab/muygps/MuyGPyS/-/commits/main)
[![Documentation Status](https://readthedocs.org/projects/muygpys/badge/?version=latest)](https://muygpys.readthedocs.io/en/latest/?badge=latest)

# Fast implementation of the MuyGPs Gaussian process hyperparameter estimation algorithm


MuyGPs is a GP estimation method that affords fast hyperparameter optimization by way of performing leave-one-out cross-validation.
MuyGPs achieves best-in-class speed and scalability by limiting inference to the information contained in k nearest neighborhoods for prediction locations for both hyperparameter optimization and tuning.
This feature affords the optimization of hyperparameters by way of leave-one-out cross-validation, as opposed to the more expensive loglikelihood evaluations requires by similar sparse methods. 


## Installation


Pip installation instructions:
```
$ pip install muygpys
```

To install from source, follow these instructions:
```
$ git clone git@github.com:LLNL/MuyGPyS.git
$ pip install -e MuyGPyS
```


## Building Docs

Automatically-generated documentation can be found at [readthedocs.io](https://muygpys.readthedocs.io/en/latest/?).

Doc building instructions:
```
$ cd /path/to/this/repo/docs
$ pip install -r requirements.txt
$ sphinx-build -b html docs docs/_build/html
```
Then open the file `docs/_build/html/index.html` in your browser of choice.


## The Basics


### Data format


`MuyGPyS` expects that each train or test observation corresponds to a row index in feature and response matrices.
In our examples we assume that train data is bundled into a `(train_count, feature_count)` feature matrix `train_features` and a `(train_count, response_count)` response matrix `train_responses`. 
In classification examples we will instead refer to a `(train_count, class_count)` label matrix `train_labels` whose rows are one-hot encodings.
Our examples will assume that the data is accessible via imaginary getter functions. 


### Constructing Nearest Neighbor Lookups


`MuyGPyS.neighbors.NN_Wrapper` is an api for tasking several KNN libraries with the construction of lookup indexes that empower fast training and inference.
The wrapper constructor expects the training features, the number of nearest neighbors, and a method string specifying which algorithm to use, as well as any additional kwargs used by the methods.
Currently supported implementations include [exact KNN using sklearn](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.NearestNeighbors.html) ("exact") and [approximate KNN using hnsw](https://github.com/nmslib/hnswlib) ("hnsw").

Construct exact and approximate  KNN data example with k = 10
```
>>> from MuyGPyS.neighors import NN_Wrapper 
>>> train_features = load_train_features()  # imaginary getter
>>> nn_count = 10
>>> exact_nbrs_lookup = NN_Wrapper(train_features, nn_count, nn_method="exact", algorithm="ball_tree")
>>> approx_nbrs_lookup = NN_Wrapper(train_features, nn_count, nn_method="hnsw", space="l2", M=16)
```

These lookup data structures are then usable to find nearest neighbors of queries in the training data.


### Sampling Batches of Data


MuyGPyS includes convenience functions for sampling batches of data from existing datasets.
These batches are returned in the form of row indices, both of the sampled data as well as their nearest neighbors.
Also included is the ability to sample "balanced" batches, where the data is partitioned by class and we attempt to sample as close to an equal number of items from each class as is possible. 

Sampling random and balanced (for classification) batches of 100 elements:
```
>>> from MuyGPyS.optimize.batch import sample_batch, get_balanced_batch
>>> train_labels = load_train_labels()  # imaginary getter
>>> batch_count = 200
>>> train_count, _ = train_features.shape
>>> batch_indices, batch_nn_indices = sample_batch(
...         exact_nbrs_lookup, batch_count, train_count
... )
>>> train_lookup = np.argmax(train["output"], axis=1)
>>> balanced_indices, balanced_nn_indices = get_balanced_batch(
...         exact_nbrs_lookup, train_lookup, batch_count
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

One sets hyperparameters such as `eps`, `sigma_sq`, as well as kernel-specific hyperparameters, e.g. `nu` and  `length_scale` for the Matern kernel, at initialization as above.
All hyperparameters other than `sigma_sq` are assumed to be fixed unless otherwise specified.

MuyGPyS depends upon linear operations on specially-constructed tensors in order to efficiently estimate GP realizations.
Constructing these tensors depends upon the nearest neighbor index matrices that we described above.
We can construct a distance tensor coalescing all of the square pairwise distance matrices of the nearest neighbors of a batch of points.
This snippet constructs a Euclidean distance tensor.
```
>>> from MuyGPyS.gp.distance import pairwise_distances
>>> pairwise_dists = pairwise_distances(
...         train_features, batch_nn_indices, metric="l2"
... )
```

We can similarly construct a matrix coalescing all of the distance vectors between the same batch of points and their nearest neighbors.
```
>>> from MuyGPyS.gp.distance import crosswise_distances
>>> crosswise_dists = crosswise_distances(
...         train_features,
...         train_features,
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
...         train_labels,
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
...         train_features,
...	    test_features,
...	    train_labels,
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
Note that this is the default behavior, and `sigma_sq` is the only hyperparameter assumed to be a training target by default. 
```
>>> from MuyGPyS.examples.regress import make_regressor
>>> train_features, train_responses = load_train()  # imaginary train getter
>>> test_features, test_responses = load_test()  # imaginary test getter
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
...         train_features,
...         train_responses,
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
>>> train_features, train_labels = load_train()  # imaginary train getter
>>> nn_kwargs = {"nn_method": "exact", "algorithm": "ball_tree"}
>>> k_kwargs = {
...         "kern": "rbf",
...         "metric": "F2",
...         "eps": {"val": 1e-5},
...         "nu": {"val": 0.38, "bounds": (0.1, 2.5)},
...         "length_scale": {"val": 7.2},
... }
>>> muygps, nbrs_lookup = make_classifier(
...         train_features,
...         train_labels,
...         nn_count=40,
...         batch_size=500,
...         loss_method="log",
...         k_kwargs=k_kwargs,
...         nn_kwargs=nn_kwargs, 
...         verbose=False,
... )    
```


### Multivariate Models


MuyGPyS also supports multivariate models via the `MuyGPyS.gp.muygps.MultivariateMuyGPS` class, which maintains a separte kernel function for each response dimension.
This class is similar in interface to `MuyGPyS.gp.muygps.MuyGPS`, but requires a list of hyperparameter dicts at initialization.
See the following example:
```
>>> from MuyGPyS.gp.muygps import MultivariateMuyGPS as MMuyGPS
>>> k_args = [
... 	    {
...                 "eps": {"val": 1e-5},
...                 "nu": {"val": 0.38, "bounds": (0.1, 2.5)},
...                 "length_scale": {"val": 7.2},
...	    },
... 	    {
...                 "eps": {"val": 1e-5},
...                 "nu": {"val": 0.67, "bounds": (0.1, 2.5)},
...                 "length_scale": {"val": 7.2},
...	    },
... ]
>>> mmuygps = MMuyGPS("matern", **k_args)
```

Training is similar, and depends upon the same neighbors index datastructures as the singular models.
In order to train, one need only loop over the models contained within the multivariate object.
```
>>> from MuyGPyS.optimize.chassis.scipy_optimize_from_indices
>>> for i, model in mmuygps.models:
>>>         scipy_optimize_from_indices(
...                 model,
...                 batch_indices,
...                 batch_nn_indices,
...                 train_features,
...	            test_features,
...	            train_responses[:, i].reshape(train_count, 1),
...                 loss_method="mse",
...                 verbose=False,
...         )
```

We also support one-line make functions for regression and classification:
```
>>> from MuyGPyS.examples.regress import make_multivariate_regressor
>>> train_features, train_responses = load_train()  # imaginary train getter
>>> nn_kwargs = {"nn_method": "exact", "algorithm": "ball_tree"}
>>> k_args = [
... 	    {
...                 "eps": {"val": 1e-5},
...                 "nu": {"val": 0.38, "bounds": (0.1, 2.5)},
...                 "length_scale": {"val": 7.2},
...	    },
... 	    {
...                 "eps": {"val": 1e-5},
...                 "nu": {"val": 0.67, "bounds": (0.1, 2.5)},
...                 "length_scale": {"val": 7.2},
...	    },
... ]
>>> muygps, nbrs_lookup = make_multivariate_regressor(
...         train_features,
...         train_responses,
...         nn_count=40,
...         batch_size=500,
...         loss_method="mse",
...	    kern="matern",
...         k_args=k_args,
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
...         train_features, batch_nn_indices, metric="l2"
... )
>>> crosswise_dists = crosswise_distances(
...         test_features,
...         train_features,
...         indices,
...         nn_indices,
...         metric='l2',
... )
>>> K = muygps.kernel(pairwise_dists)
>>> Kcross = muygps.kernel(crosswise_dists)
>>> predictions = muygps.regress(K, Kcross, train_responses[nn_indices, :])
```

Again if you do not want to reuse your tensors, you can run the more compact:
```
>>> indices = np.arange(test_count)
>>> nn_indices = train_nbrs_lookup.get_nns(test["input"])
>>> muygps.regress_from_indices(
...         indices,
...	    nn_indices,
...	    test_features,
...	    train_features,
...	    train_responses,
... )
```

Multivariate models support the same functions.

More complex workflows are of course available.
See the `MuyGPyS.examples` high-level API functions for examples.


## API Examples


Listed below are several examples using the high-level APIs located in `MuyGPyS.examples.classify` and `MuyGPyS.examples.regress`.
Note that one need not go through these APIs to use `MuyGPyS`, but they condense many basic workflows into a single function call.
In all of these examples, note that if all of the hyperparameters are fixed in `k_kwargs` (i.e. you supply no optimization bounds), the API will perform no optimization and will instead simply predict on the data problem using the provided kernel.
While these examples all use a single model, one can modify those with multivariate responses to use multivariate models by supplying the additional keyword argument `kern=kernel_name`, for `kernel_name in ['rbf', 'matern']` and providing a list of hyperparameter dicts to the keyword argument `k_kwargs` as above.

## Regression


The following example performs GP regression on the [Heaton spatial statistics case study dataset](https://github.com/finnlindgren/heatoncomparison).
In the example, `load_heaton` is a unspecified function that reads in the dataset in the specified dict format.
In practice, a user can use any conforming dataset.
If one wants to predict on a univariate response as in this example, one must ensure the data is stored as a matrix rather than as a vector, i.e. that `train['output'].shape = (train_count, 1)`.
The regression API adds a `sigma_sq` scale parameter for the variance.
One can set `sigma_sq` using the `hyper_dict` kwarg like other hyperparameters.
The API expects that `sigma_sq` is a `numpy.ndarray` with a value associated with each dimension of the response, i.e. that `train['output'].shape[1] == len(sigma_sq)`.
In general, one should only manually set `sigma_sq` if they are certain they know what they are doing. 

Regress on Heaton data with no variance
```
>>> import numpy as np
>>> from MuyGPyS.examples.regress import do_regress
>>> from MuyGPyS.optimize.objective import mse_fn
>>> train_features, train_responses = load_heaton_train()  # imaginary train getter
>>> test_features, test_responses = load_heaton_test()  # imaginary test getter
>>> nn_kwargs = {"nn_method": "exact", "algorithm": "ball_tree"}
>>> k_kwargs = {
...         "kern": "rbf",
...         "metric": "F2",
...         "eps": {"val": 1e-5},
...         "nu": {"val": 0.38, "bounds": (0.1, 2.5)},
...         "length_scale": {"val": 7.2},
... }
>>> muygps, nbrs_lookup, predictions = do_regress(
...         test_features,
...         train_features,
...         train_responses,
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
>>> print(f"mse : {mse_fn(predictions, test_responses)}")
obtains mse: 2.345136495565052
```

If one requires the (individual, independent) posterior variances for each of the predictions, one can pass `variance_mode="diagonal"`.
This mode assumes that each output dimension uses the same model, and so will output an additional vector `variance` with a scalar posterior variance associated with each test point.
The API also returns a (possibly trained) `MuyGPyS.gp.MuyGPS` or `MuyGPyS.gp.MultivariateMuyGPS` instance, whose `sigma_sq` member reports an array of multiplicative scaling parameters associated with the variance of each dimension.
Obtaining the tuned posterior variance implies multiplying the returned variance by the scaling parameter along each dimension.


Regress on Heaton data while estimating diagonal variance
```
>>> import numpy as np
>>> from MuyGPyS.examples.regress import do_regress
>>> from MuyGPyS.optimize.objective import mse_fn
>>> train_features, train_responses = load_heaton_train()  # imaginary train getter
>>> test_features, test_responses = load_heaton_test()  # imaginary test getter
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
...         test_features,
...         train_features,
...         train_responses,
...         nn_count=30,
...         batch_size=200,
...         loss_method="mse",
...         variance_mode="diagonal",
...         k_kwargs=k_kwargs,
...         nn_kwargs=nn_kwargs,
...         verbose=False,
... )
>>> print(f"mse : {mse_fn(predictions, test_responses)}")
obtains mse: 2.345136495565052
>>> print(f"diagonal posterior variance: {variance * muygps.sigma_sq()}")
diagonal posterior variance: [0.52199482 0.45934382 0.81381388 ... 0.64982631 0.45958342 0.68602048]
```

Independent diagonal variance for each test item is the only form of posterior variance supported for a single model, and independent diagonal variance for each test item along each response dimension is the only form of posterior variacne supported for a multivariate model. 
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
>>> from MuyGPyS.examples.two_class_classify_uq import do_classify_uq, do_uq, example_lambdas
>>> from MuyGPyS.optimize.objective import mse_fn
>>> train_features, train_labels = load_stargal_train()  # imaginary train getter
>>> test_features, test_labels = load_stargal_test()  # imaginary test getter
>>> nn_kwargs = {"nn_method": "exact", "algorithm": "ball_tree"}
>>> k_kwargs = {
...         "kern": "rbf",
...         "metric": "F2",
...         "eps": {"val": 1e-5},
...         "nu": {"val": 0.38, "bounds": (0.1, 2.5)},
...         "length_scale": {"val": 7.2},
... }
>>> muygps, nbrs_lookup, surrogate_predictions, masks = do_classify_uq(
...         test_features,
...         train_features,
...         train_labels,
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
>>> accuracy, uq = do_uq(surrogate_predictions, test_labels, masks)
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

If uncertainty quantification is not desired, or the classifcation problem in question involves more than two classes, see instead an example workflow like that in `MuyGPyS.examples.classify.do_classify`.

Run MNIST without UQ example instructions:
```
>>> import numpy as np
>>> from MuyGPyS.examples.classify import do_classify
>>> from MuyGPyS.optimize.objective import mse_fn
>>> train_features, train_labels = load_stargal_train()  # imaginary train getter
>>> test_features, test_labels = load_stargal_test()  # imaginary test getter
>>> nn_kwargs = {"nn_method": "exact", "algorithm": "ball_tree"}
>>> k_kwargs = {
...         "kern": "rbf",
...         "metric": "F2",
...         "eps": {"val": 1e-5},
...         "nu": {"val": 0.38, "bounds": (0.1, 2.5)},
...         "length_scale": {"val": 7.2},
... }
>>> muygps, nbrs_lookup, surrogate_predictions = do_classify(
...         test_features,
...         train_features,
...         train_labels,
...         nn_count=30,
...         batch_size=200,
...         loss_method="log",
...         variance_mode=None,
...         k_kwargs=k_kwargs,
...         nn_kwargs=nn_kwargs,
...         verbose=False,
... )
>>> predicted_labels = np.argmax(surrogate_predictions, axis=1)
>>> true_labels = np.argmax(test_labels, axis=1)
>>> accuracy = np.mean(predicted_labels == true_labels)
>>> print(f"obtained accuracy: {accuracy}")
0.97634
```


# About

## Authors

* Benjamin W. Priest (priest2 at llnl dot gov)
* Amanada L. Muyskens (muyskens1 at llnl dot gov)

## Papers

MuyGPyS has been used the in the following papers (newest first):

1. [Gaussian Process Classification fo Galaxy Blend Identification in LSST](https://arxiv.org/abs/2107.09246)
2. [Star-Galaxy Image Separation with Computationally Efficient Gaussian Process Classification](https://arxiv.org/abs/2105.01106)
3. [Star-Galaxy Separation via Gaussian Processes with Model Reduction](https://arxiv.org/abs/2010.06094)

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

## License

MuyGPyS is distributed under the terms of the MIT license.
All new contributions must be made under the MIT license.

See [LICENSE-MIT](LICENSE-MIT), [NOTICE](NOTICE), and [COPYRIGHT](COPYRIGHT) for details.

SPDX-License-Identifier: MIT

## Release

LLNL-CODE-824804
