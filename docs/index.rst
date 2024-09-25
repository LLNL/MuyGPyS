.. MuyGPyS documentation master file, created by
   sphinx-quickstart on Wed Jul 14 12:12:25 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

MuyGPyS (|version|) Reference
=========================================

MuyGPyS is toolkit for training approximate Gaussian Process (GP) models using the MuyGPs (Muyskens, Goumiri, Priest, Schneider) algorithm. 


Citation
========

If you use MuyGPyS in a research paper, please reference our article::

  @article{muygps2021,
    title={MuyGPs: Scalable Gaussian Process Hyperparameter Estimation Using Local Cross-Validation},
    author={Muyskens, Amanda and Priest, Benjamin W. and Goumiri, Im{\`e}ne and Schneider, Michael},
    journal={arXiv preprint arXiv:2104.14581},
    year={2021}
  }



.. toctree::
   :maxdepth: 2
   :caption: Package Documentation:

   MuyGPyS/neighbors
   MuyGPyS/gp
   MuyGPyS/optimize
   MuyGPyS/examples
   MuyGPyS/torch

.. toctree::
   :maxdepth: 2
   :caption: Examples:

   examples/univariate_regression_tutorial.ipynb
   examples/neighborhood_illustration.ipynb
   examples/torch_tutorial.ipynb
   examples/fast_regression_tutorial.ipynb
   examples/anisotropic_tutorial.ipynb
   examples/loss_tutorial.ipynb

Variable Name Conventions
=========================

We make use of several canonical variable names that refer to tensor shape
dimensions.
Here is a partial list of the major names and their meanings.

* `train_count` - the number of training observations.
* `test_count` - the number of test or prediction observations.
* `batch_count` - the number of elements to be predicted. Can coincide with `train_count` or `test_count` depending on usage. Sometimes also called `data_count`.
* `feature_count` - the number of features in the observations. Omitted for univariate feature spaces.
* `response_count` - the number of response variables. Omitted for univariate responses.
* `nn_count` - the number of nearest neighbors upon which predictions are conditioned.
* `out_shape` - a tuple referring to the shape associated with the output shape of the cross-covariance. For a univariate problem, `in_shape = (nn_count,)`. For a multivariate problem, `out_shape` most likely refers to `(nn_count, response_count)`.
* `in_shape` - a tuple referring to the shape associated with how the covariance is conditioned on observations. For a univariate problem, `in_shape == (nn_count,)`. For a multivariate problem, `in_shape` might refer to `(nn_count, response_count)`, but could instead have a different second element if the observations do not come from the same space as the predictions.

.. toctree::
   :maxdepth: 2
   :caption: Resources:

   resources/references

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
