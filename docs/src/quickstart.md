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

We can chain several operators together through multiple ways 

### Compose Operator

Consider two matrices:

```julia
L = ParMatrix(10, 4)
R = ParMatrix(10, 4)
```

We can now chain and parametrize them by:

```julia
C = L * R'
θ = init(C)
```

This allows us to perform several operations such as

```julia

using Zygote
using LinearAlgebra

x = rand(10)
C(θ) * x
gradient(θ -> norm(C(θ) * x), θ)
```

without ever constructing the full matrix `LR'`, a method more popularly referred to as [LR-decomposition](https://link.springer.com/chapter/10.1007/978-3-662-65458-3_11).

### Kronecker Operator

[Kronecker Product](https://en.wikipedia.org/wiki/Kronecker_product) is a most commonly used to represent the outer product on 2 matrices.

We can use this to describe linearly separable transforms that act along different dimensions on a given input tensor.

For example, consider the following tensor:

```julia
T = Float32
x = rand(T, 10, 20, 30)
```

We now define the transformation that would act along each dimension. In this case, a [Fourier Transform](https://en.wikipedia.org/wiki/Fourier_transform).

#### Fourier Transform Example
```julia
Fx = ParDFT(T, 10)
Fy = ParDFT(Complex{T}, 20)
Fz = ParDFT(Complex{T}, 30)
```

We can now chain them together using a Kronecker Product:

```julia
F = Fz ⊗ Fy ⊗ Fx
```

Now, we can compute this action on our input by simply doing:

```julia
F * vec(x)
```

!!! tip "This can be extended to any parametrized operators"
    For example, in order to apply a linear transform along the y, z dimension while performing no operation along x, one can do:
    ```julia
        Sx = ParIdentity(T, 10)
        Sy = ParMatrix(T, 20, 20)
        Sz = ParMatrix(T, 30, 30)

        S = Sz ⊗ Sy ⊗Sx
        θ = init(S)

        S(θ) * vec(x)
    ```
