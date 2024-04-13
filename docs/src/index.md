### ParametricOperators.jl

Modern machine learning and scientific computing is increasingly interested in the idea of tensor programs, or programs where the fundamental object is the tensor: a multidimensional array of numbers.

`ParametricOperators.jl` is an abstraction based on Kronecker products, designed for manipulating large scale tensors. It utilizes lazy operators to facilitate distribution and gradient computation effectively.

!!! note "Example usage of ParametricOperators.jl"
    [`ParametricDFNOs.jl`](https://github.com/slimgroup/ParametericDFNOs.jl) is a library built on top of [`ParametricOperators.jl`](https://github.com/slimgroup/ParametricOperators.jl) that allows for large scale machine learning using Fourier Neural Operators (FNOs)

Read our paper [here](https://www.youtube.com/watch?v=xvFZjo5PgG0).

### Authors

This package is developed and maintained by Felix J. Herrmann's [SlimGroup](https://slim.gatech.edu/) at Georgia Institute of Technology. The main contributors of this package are:

- [Richard Rex](https://www.linkedin.com/in/richard-rex/)
- Thomas Grady
- Mark Gliens

### License

```
MIT License

Copyright (c) 2024 SLIM Group @ Georgia Institute of Technology

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```
