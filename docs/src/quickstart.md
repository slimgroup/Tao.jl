## Installation

Add `ParametricOperators.jl` as a dependency to your environment.

To add, either do:

```julia
julia> ]
(v1.9) add ParametricOperators
```

OR

```julia
julia> using Pkg
julia> Pkg.activate("path/to/your/environment")
julia> Pkg.add("ParametricOperators")
```

## Simple Operator

Make sure to include the package in your environment

```julia
using ParametricOperators
```

Lets start by defining a Matrix Operator of size `10x10`:

```julia
A = ParMatrix(10, 10)
```

Now, we parametrize our operator with some weights `θ`:

```julia
θ = init(A)
```

We can now apply our operator on some random input:

```julia
x = rand(10)
A(θ) * x
```

## Gradient Computation

!!! note "Limited AD support"
    Current support only provided for Zygote.jl

Make sure to include an AD package in your environment

```julia
using Zygote
```

Using the example above, one can find the gradient of the weights or your input w.r.t to some objective using a standard AD package:

```julia
# Gradient w.r.t weights
θ′ = gradient(θ -> sum(A(θ) * x), θ)

# Gradient w.r.t input
x′ = gradient(x -> sum(A(θ) * x), x)
```

## Chaining Operators

## Kronecker Operator

## Distributing Kronecker Operator
