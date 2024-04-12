### ParametricOperators.jl

Modern machine learning and scientific computing is increasingly interested in the idea of tensor programs, or programs where the fundamental object is the tensor: a multidimensional array of numbers.

`ParametricOperators.jl` is an abstraction based on Kronecker products, designed for manipulating large scale tensors. It utilizes lazy operators to facilitate distribution and gradient computation effectively.

!!! note "Example usage of ParametricOperators.jl"
    [`ParametricDFNOs.jl`](https://github.com/slimgroup/ParametericDFNOs.jl) is a library built on top of [`ParametricOperators.jl`](https://github.com/slimgroup/ParametricOperators.jl) that allows for large scale machine learning using Fourier Neural Operators (FNOs)

Read our paper [here](https://www.youtube.com/watch?v=dQw4w9WgXcQ).
