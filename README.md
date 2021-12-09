[![pipeline status](https://lc.llnl.gov/gitlab/muygps/MuyGPyS/badges/main/pipeline.svg)](https://lc.llnl.gov/gitlab/muygps/MuyGPyS/-/commits/main)
[![Documentation Status](https://readthedocs.org/projects/muygpys/badge/?version=latest)](https://muygpys.readthedocs.io/en/latest/?badge=latest)

# Fast implementation of the MuyGPs Gaussian process hyperparameter estimation algorithm


MuyGPs is a GP estimation method that affords fast hyperparameter optimization 
by way of performing leave-one-out cross-validation.
MuyGPs achieves best-in-class speed and scalability by limiting inference to the information contained in k nearest neighborhoods for prediction locations for 
both hyperparameter optimization and tuning.
This feature affords the optimization of hyperparameters by way of leave-one-out cross-validation, as opposed to the more expensive loglikelihood evaluations 
required by similar sparse methods. 


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

Additionally check out the develop branch to access the latest features in 
between stable releases.

## Building Docs

Automatically-generated documentation can be found at 
[readthedocs.io](https://muygpys.readthedocs.io/en/latest/?).

Doc building instructions:
```
$ cd /path/to/this/repo/docs
$ pip install -r requirements.txt
$ sphinx-build -b html docs docs/_build/html
```
Then open the file `docs/_build/html/index.html` in your browser of choice.


## Tutorials and Examples 

Our documentation includes several jupyter notebook tutorials at 
`docs/examples`.
These tutorials are also include in the 
[online documentation](https://muygpys.readthedocs.io/en/latest/?).

See in particular the 
[univariate regression tutorial](docs/examples/univariate_regression_tutorial.ipynb)
for a low-level introduction to the use of `MuyGPyS`.
See also the 
[regression api tutorial](docs/examples/regress_api_tutorial.ipynb)
describing how to coalesce the same simple workflow into a one-line call.


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
