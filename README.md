# ParametricOperators.jl

[![][license-img]][license-status]
[![Documenter](https://github.com/slimgroup/ParametricOperators.jl/actions/workflows/Documenter.yml/badge.svg)](https://github.com/slimgroup/ParametricOperators.jl/actions/workflows/Documenter.yml)
[![TagBot](https://github.com/slimgroup/ParametricOperators.jl/actions/workflows/TagBot.yml/badge.svg)](https://github.com/slimgroup/ParametricOperators.jl/actions/workflows/TagBot.yml)

<!-- [![][zenodo-img]][zenodo-status] -->

ParametricOperators.jl is a Julia Language-based scientific library designed to facilitate the creation and manipulation of tensor operations involving large-scale data using Kronecker products. It provides an efficient and mathematically consistent way to express tensor programs and distribution in the context of machine learning.

## Features
- <b>Kronecker Product Operations:</b> Implement tensor operations using Kronecker products for linear operators acting along multiple dimensions.
- <b>Parameterization Support:</b> Enables parametric functions in tensor programs, crucial for statistical optimization algorithms.
- <b>High-Level Abstractions:</b> Close to the underlying mathematics, providing a seamless user experience for scientific practitioners.
- <b>Distributed Computing:</b> Scales Kronecker product tensor programs and gradient computation to multi-node distributed systems.
- <b>Domain-Specific Language:</b> Optimized for Julia's just-in-time compilation, allowing for the construction of complex operators entirely at compile time.

## Setup

   ```julia
   julia> using Pkg
   julia> Pkg.activate("path/to/your/environment")
   julia> Pkg.add("ParametricOperators")
   ```

This will add `ParametricOperators.jl` as dependency to your project

## Documentation

Check out the [Documentation](https://slimgroup.github.io/ParametricOperators.jl) for more or you get started by running some [examples](https://github.com/turquoisedragon2926/ParametricOperators.jl-Examples)!

## Issues

This section will contain common issues and corresponding fixes. Currently, we only provide support for Julia-1.9

## Authors

Richard Rex, [richardr2926@gatech.edu](mailto:richardr2926@gatech.edu) <br/>
Thomas Grady <br/>
Mark Glines <br/>

[license-status]:LICENSE
<!-- [zenodo-status]:https://doi.org/10.5281/zenodo.6799258 -->
[license-img]:http://img.shields.io/badge/license-MIT-brightgreen.svg?style=flat?style=plastic
<!-- [zenodo-img]:https://zenodo.org/badge/DOI/10.5281/zenodo.3878711.svg?style=plastic -->
