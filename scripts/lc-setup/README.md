# Environment Setup on LC for NVidia GPU support

As specified in the top-level README, it is necessary to set up your environment
variables so that JAX can find the installed CUDA and CuDNN binaries.
On LC, these binaries are not installed in the default location, we need to 
explicitly export them.
The bash files in this folder will do so for the corresponding resource.
It is a good idea to source the correct file in your `.profile` or equivalent
(or simply copy the export lines into your `.profile`) so that your environment
will be automatically configured at shell launch time. 