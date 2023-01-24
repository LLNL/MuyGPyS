# Testing bed for torch implementation of MuyGPs

This directory contains tests meant specifically for the torch implementation of MuyGPs.
These tests must be run using the torch backend, which can be turned on using
```
$ export MUYGPYS_BACKEND=torch
``` 
in the user's shell environment. 

If setting environment variables is impractical, one can also use the following
workflow. 

```
from MuyGPyS import config
MuyGPyS.config.update("muygpys_backend","torch")

...subsequent imports from MuyGPyS
```