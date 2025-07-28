# How to Contribute

`MuyGPyS` is an open source project.
Our team welcomes contributions from collaborators in the form of raising issues 
as well as code contributions including hotfixes, code improvements, and new
features.

`MuyGPyS` is distributed under the terms of the 
[MIT license](https://github.com/LLNL/MuyGPyS/blob/develop/LICENSE-MIT). 
All new contributions must be made under this license.

If you identify a problem such as a bug or awkward or confusing code, or require
a new feature, please feel free to start a thread on our 
[issue tracker](https://github.com/LLNL/MuyGPyS/issues).
Please first review the existing issues to avoid duplicates. 
 
If you plan on contributing to `MuyGPyS`, please review the 
[issue tracker](https://github.com/LLNL/MuyGPyS/issues) to check for threads 
related to your desired contribution. 
We recommend creating an issue prior to issuing a pull request if you are 
planning significant code changes or have questions. 

# Contribution Workflow

These guidelines assume that the reader is familiar with the basics of 
collaborative development using git and GitHub.
This section will walk through our preferred pull request workflow for 
contributing code to `MuyGPyS`. 
The tl;dr guidance is:
- Fork the [LLNL MuyGPyS repository](https://github.com/LLNL/MuyGPyS)
- Create a descriptively named branch 
(`feature/myfeature`, `iss/##`, `hotfix/bugname`, etc) in your fork off of 
the `develop` branch
- Commit code, following our [guidelines](#formatting-guidelines)
- Create a [pull request](https://github.com/LLNL/MuyGPyS/compare/) from your 
branch targeting the LLNL `develop` branch

## Forking MuyGPyS

If you are not a `MuyGPyS` developer at LLNL, you will not have the necessary
permissions to push new branches to the repository.
Even `MuyGPyS` developers at LLNL will want to use forks for most contributions.
This will create a clean copy of the repository that you own, and will allow for 
exploration and experimentation without muddying the history of the central 
repository.

If you intend to maintain a persistent fork of `MuyGPyS`, it is a best practice 
to set the LLNL repository as the `upstream` remote in your fork. 
```
$ git clone git@github.com:your_name/MuyGPyS.git
$ cd MuyGPyS
$ git remote add upstream git@github.com:LLNL/MuyGPyS.git
```
This will allow you to incorporate changes to the `main` and `develop` 
branches as they evolve.
For example, perform the following commands to update your fork's develop
branch:
```
$ git fetch upstream
$ git checkout develop
$ git pull upstream develop
$ git push origin develop
```
It is important to keep your develop branch up-to-date to reduce merge conflicts
resulting from future PRs.

## Contribution Types

Most contributions will fit into one of the follow categories, which by 
convention should be committed to branches with descriptive names.
Here are some examples:
- A new feature (`feature/<feature-name>`)
- A bug or hotfix (`hotfix/<bug-name>` or `hotfix/<issue-number>`)
- A response to a [tracked issue](https://github.com/LLNL/MuyGPyS/issues) 
(`iss/<issue-number>`)
- A work in progress, not to be merged for some time (`wip/<change-name>`)

### Developing a new feature

New features should be based on the develop branch:
```
$ git checkout develop
$ git pull upstream develop
```
You can then create new local and remote branches on which to develop your 
feature.
```
$ git checkout -b feature/<feature-name>
$ git push --set-upstream origin feature/<feature-name>
```
Commit code changes to this branch, and add tests to the `tests` directory that
validate the correctness of your code, modifying existing tests if need be.
Be sure that your test runs successfully.

Make sure that you follow our [formatting guidelines](#formatting-guidlines) for 
any changes to the source code or build system. 
If you create new methods or classes, please add ReStructuredText documentation
and make sure that the documentation builds locally.
If you create additional documentation notebooks, or modify existing ones,
please ensure that you use `Kernel > Restart and Clear Output` and save your
local version of the notebook before committing so that you do not accidentally
commit execution state to the git history.

Once your feature is complete and your tests are passing, ensure that your 
remote fork is up-to-date and 
[create a PR](https://github.com/LLNL/MuyGPyS/compare). 

### Developing a hotfix

Firstly, please check to ensure that the bug you have found has not already been
fixed in `develop`. 
If it has, we suggest that you temporarily swap your environment to use the
`develop` branch or ask for a new release.

If you have identified an unsolved bug, you can document the problem and create
an [issue](https://github.com/LLNL/MuyGPyS/issues).
If you would like to solve the bug yourself, follow a similar protocol to 
feature development.
First, ensure that your fork's `develop` branch is up-to-date.
```
$ git checkout develop
$ git pull upstream develop
```
You can then create new local and remote branches on which to write your bug 
fix.
```
$ git checkout -b hotfix/<bug-name>
$ git push --set-upstream origin hotfix/<bug-name>
```

Firstly, create a test added to the `tests` directory that reproduces the bug or 
modify an existing test to catch the bug if that is more appropriate.
Then, modify the code to fix the bug and ensure that your new or modified test
case(s) pass. 

Please update function and class documentation to reflect any changes as 
appropriate, and follow our [formatting guidlines](#formatting-guidelines) with 
any new code.

Once you are satisfied that the bug is fixed, ensure that your remote fork is 
up-to-date and [create a PR](https://github.com/LLNL/MuyGPyS/compare). 

# Tests

<!-- `MuyGPyS` uses GitHub actions for continuous integration tests. 
Our tests run automatically against every new commit and pull request, and pull
requests must pass all tests prior to being considered for merging into the main
project. -->
Pull requests must pass all tests prior to being considered for merging into the 
main project.
If you are developing a new feature or fixing a bug, please add a test or modify
existing tests that will ensure the correctness of the new code. 

`MuyGPyS`'s tests are contained in the `tests` directory, and make use of the 
`absl` library. 
pip install MuyGPyS from source using the `tests` extras flags
(`pip install -e .[tests]`) to automatically populate your environment with all
dependencies needed to run tests.

# Formatting Guidelines

## Partitioning Math Functions

All significant math functions in `MuyGPyS` must have both pure numpy, 
JAX, MPI, and Torch implementations, which are located within `MuyGPyS._src`.
The front-end version of the functions, located within the proper subpackages,
import the appropriate function based upon the `MUYGPYS_BACKEND` environment
variable as explained in the README. 
Examples proliferate in the code.
Here is a particular example using `_mse_fn()` for illustration:

`MuyGPyS._src.optimize.loss.numpy` contains the following function:
```
def _mse_fn(
    predictions: np.ndarray,
    targets: np.ndarray,
) -> float:
    batch_count, response_count = predictions.shape
    squared_errors = np.sum((predictions - targets) ** 2)
    return squared_errors / (batch_count * response_count)
```
while `MuyGPyS._src.optimize.loss.jax` contains a JAX version
```
@jit
def _mse_fn(
    predictions: jnp.ndarray,
    targets: jnp.ndarray,
) -> float:
    batch_count, response_count = predictions.shape
    squared_errors = jnp.sum((predictions - targets) ** 2)
    return squared_errors / (batch_count * response_count)
```
There are MPI and torch versions in similarly-named files.
Note that these implementations share the same signature.
Meanwhile, API files import `_mse_fn` from `MuyGPyS._src.optimize.loss`
(irrelevant bits omitted):
```
from MuyGPyS._src.optimize.loss import _mse_fn

def mse_fn(
    predictions: np.ndarray,
    targets: np.ndarray,
) -> float:
    return _mse_fn(predictions, targets)
```
`MuyGPyS/_src/optimize/loss/__init__.py` has the following magic function call
that specifies the correct import behavior:
```
(
    _mse_fn,
    _cross_entropy_fn,
    _lool_fn,
    _lool_fn_unscaled,
    _pseudo_huber_fn,
    _looph_fn,
) = _collect_implementation(
    "MuyGPyS._src.optimize.loss",
    "_mse_fn",
    "_cross_entropy_fn",
    "_lool_fn",
    "_lool_fn_unscaled",
    "_pseudo_huber_fn",
    "_looph_fn",
)
```
All imports of math functions hit such an `__init__.py` file, which mysteriously
imports the correct function implementations based upon the `MUYGPYS_BACKEND`
environment variable. 
All significant math functions must support the same code workflow to maintain
correct backend behavior.
If a function is not possible to implement in a particular backend, still create
the function signature and have it throw a `NotImplementedError`.
There are several examples of this in the codebase.

## Commonly-used math operations and imports

Insofar as possible, please do not contribute code to the core library that
directly imports any backend math libraries, such as numpy or Torch.
Instead of, e.g., `import numpy as np`, please instead use
`import MuyGPyS._src.math as mm`.
`MuyGPyS._src.math` is intended to be an abstraction of all backend math
engines, including `numpy`, `jax.numpy`, and `torch`.
This involves some remapping of namespaces for consistency.
For example, `mm.ndarray` can refer to `numpy.ndarray`, `jax.numpy.ndarray`, or
`torch.Tensor`, depending on the state of `MUYGPYS_BACKEND` at import time.
`MuyGPyS._src.math` is used heavily throughout the code, and usage should be
clear from examples.
If it is absolutely necessary to import a specific backend, e.g., `numpy`, then
please import the wrapped version with `import MuyGPyS._src.math.numpy as np`.
This is usually only necessary for testing.

Not all backend operations are currently supported.
Add additionally functions to `MuyGPyS/_src/math/numpy.py`,
`MuyGPyS/_src/math/jax.py`, et cetera and update the corresponding invocation of
`_collect_implementation` in `MuyGPyS/_src/math/__init__.py` as needed.

## Naming Style

In general, all names in `MuyGPyS` should be as succinct as possible while 
remaining descriptive. 

MuyGPyS methods and variables should all be `snake_case`, i.e. one or more 
lowercase words separated by underscores.
Mathy variable names like `x` and `y` should be avoided in favor of descriptive 
names such as `train_features`.
This includes kernel hyperparameters.
For example the Matèrn smoothness parameter is known as `nu` in many libraries,
but we call it `smoothness`.

Here are some example function and variable names
```
def my_cool_function() -> int
    magic_number = 14
    return magic_number

def my_other_cool_function(fun_array: np.array) -> np.array
    return 2.5 * fun_array
```

MuyGPyS classes should be `PascalCase`, one or more words that start with 
uppercase letters with no separator.
What classes do exist are predominantly functors (classes organized around a
`__call__()` method) or near-functors (classes organized around two or more
related named functions, such as `MuyGPS` and its methods `MuyGPS.regress()`
and `MuyGPS.regress_from_indices()`).
Non-functor and non-near-functor classes are to be avoided where possible, as
MuyGPyS is a "soft" functional package.
Class members (variables and functions) that are intended for internal use 
should be prepended with an underscore.
Think carefully as to whether what you are trying to do really requires the 
statefulness of a functor, or whether it can be alternatively accomplished via a 
function or set of functions. 

Here is an example functor class:
```
class NeatMultiplier:
    def __init__(self, coefficient: float) -> None:
        self._coefficient = coefficient
    def __call__(self, val: float) -> float:
        return val * self._coefficient
```

## Python type hints

`MuyGPyS` uses [type hints](https://www.python.org/dev/peps/pep-0484/) for all 
function arguments and return values.
These type hints are useful for documentation, debugging, and for linting using, 
e.g. [mygpy](http://mypy-lang.org/).

The examples in [the prior section](#naming-style) include some simple examples.
Here is a slightly more sophisticated example of a functor that can return 
either a `float` or `np.ndarray` object:
```
import numpy as np
from typing import Union

class VariableMultiplier:
    def __init__(self, coefficient: float) -> None:
        self._coefficient = coefficient
    def __call__(
        self, val: Union[float, np.ndarray]
    ) -> Union[float, np.ndarray]:
        return val * self._coefficient
```

## Documentation Style

All API methods and classes that are called by users must include a 
[ReStructuredText](https://www.sphinx-doc.org/en/master/usage/restructuredtext/basics.html) 
docstring.
This docstring must include the following components:
- A single sentence summarizing its use.
- Formatted descriptions of any arguments
- Formatted descriptions of any returned objects
- Formatted descriptions of any `Error`s that could be raised and the conditions
that trigger them.

Additional components might include:
- Paragraph(s) following the single sentence summary expanding upon detail
- RST links to other sections of the package
- Hyperlinks to external references 
- Inline LaTeX formatted strings illustrating the math
- Formatted code example(s) demonstrating the functor or function's use.

*Important Detail* MuyGPyS deals with many vectors, matrices, and tensors. 
For clarity, functions and functors operating on such objects should record in 
their docstrings the expected shape of their arguments, if they are known.
Most vectors, matrices, and tensors throughout the codebase have dimensions
defined in terms of the scalar quantities `train_count`, `test_count`,
`batch_count`, or `nn_count`.

*Important Detail* The formatting required for functions with multiple return
values (i.e. functions that return a tuple of objects) differs from that of 
functions with a single return object. 
Here is an example function with a single return value:
```
def array_multiplier(fun_array: np.array, coeff: float = 2.5) -> np.array
    """
    Multiplies an array by a scalar.

    .. math::
        Y = cX
    
    Example:
        >>> import numpy as np
        >>> X = np.random.randn(10, 5)
        >>> c = 2.5
        >>> Y = array_multiplier(X, coeff=c)

    Args:
        fun_array:
            A tensor of shape `(shape_1, shape_2, shape_3)`
        coeff:
            A floating point coefficient.
    
    Returns:
        A tensor of shape `(shape_1, shape_2, shape_3)`, elementwise multiplied
        by the provided scalar coefficient.
    """
    return coeff * fun_array
```
Meanwhile, here is a similar function with multiple return values.
Note the different formatting of the returns.
```
from typing import Tuple

def verbose_array_multiplier(
  fun_array: np.array, coeff: float = 2.5
) -> Tuple[np.array, float]
    """
    Multiplies an array by a scalar, and also returns the scalar + 1.

    .. math::
        Y = cX
    
    Example:
        >>> import numpy as np
        >>> X = np.random.randn(10, 5)
        >>> c = 2.5
        >>> Y, c_prime = array_multiplier(X, coeff=c)

    Args:
        fun_array:
            A tensor of shape `(shape_1, shape_2, shape_3)`
        coeff:
            A floating point coefficient.
    
    Returns
    -------
        ret_array:
            A tensor of shape `(shape_1, shape_2, shape_3)`, elementwise 
            multiplied by the provided scalar coefficient.
        ret_coeff:
            The provided scalar coefficient incremented by one.
    """
    ret_array = coeff * fun_array
    ret_coeff = coeff + 1
    return ret_array, ret_coeff
```
See the codebase for more sophisticated examples.

## Code Style

`MuyGPyS` uses the [black](https://pypi.org/project/black/) formatter to 
guarantee a consistent format for python code.
`black` is easy to use, and can be easily instrumented to auto-format
code using most modern editors.
The only custom setting used by MuyGPyS is `--line-length 80`, which can be
added to your auto-formatting infrastructure if desired.

## Linting

`MuyGPyS` uses the [flake8](https://flake8.pycqa.org/en/latest/) linter to
enforce several formatting rules in its CI.
To run the checks locally from the repo roots, simply run
`sh scripts/flake8/*`.
