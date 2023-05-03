.. MuyGPyS documentation master file, created by
   sphinx-quickstart on Wed Jul 14 12:12:25 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

MuyGPyS Reference
===================================

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
   examples/fast_regression_tutorial.ipynb
   examples/torch_tutorial.ipynb

.. toctree::
   :maxdepth: 2
   :caption: Resources:

   resources/references

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
