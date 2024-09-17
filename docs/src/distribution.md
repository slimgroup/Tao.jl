# Distribution as Linear Algebra

We adapt an approach of looking at distribution of tensor computation as Linear Algebra operations. 

This allows `ParametricOperators.jl` to offer several high level API in order to perform controlled parallelism as part of your tensor program in the context of machine learning.

## Kronecker Distribution

### Distributed Fourier Transform

Let's consider the example of Fourier Transform as seen in the [Fourier Transform Example](@ref) 

```julia
# Define type and the size of our global tensor
T = Float32
gx, gy, gz = 10, 20, 30

Fx = ParDFT(T, gx)
Fy = ParDFT(Complex{T}, gy)
Fz = ParDFT(Complex{T}, gz)

F = Fz ⊗ Fy ⊗ Fx
```

Assume that our data is partitioned across multiple machine according to the following scheme:

```julia
partition = [1, 1, 2]
```

Each element of `partition` denotes the number of processing elements that divide our input tensor along that dimension.

For eg. given the above partition and global size, our local tensor would be of size:

```julia
x = rand(T, 10, 20, 15)
```

OR in other terms:

```julia
localx, localy, localz = [gx, gy, gz] .÷ partition
x = rand(T, localx, localy, localz)
```

Now, following the method seen in several recent works (Grady et al., [2022](https://arxiv.org/pdf/2204.01205.pdf)) and [traditional distributed FFTs](https://jipolanco.github.io/PencilFFTs.jl/dev/tutorial/), we can distribute the application of our linearly separable transform across multiple processing elements by simply doing:

```julia
F = distribute(F, partition)
```

Now, to apply the Fourier Transform to our tensor, one can do:

```julia
F * vec(x)
```

Another out-of-box example can be seen at [Distributed FFT of a 3D Tensor](@ref)

### Distributed Convolution

!!! note "Definition of Convolution"
    Convolution here refers to the application of a linear transform along the channel dimension

Now, in order to extend this to a convolution layer, lets consider the following partitioned tensor:

```julia
T = Float32

gx, gy, gc = 10, 30, 50
partition = [2, 2, 1]

nx, ny, nc = [gx, gy, gc] .÷ partition
x = rand(T, nx, ny, nc)
```

Our tensor is sharded across x and y dimensions by 2 processing element along each dimension. 

We can define the operators of our convolution as:

```julia
Sx = ParIdentity(T, gx)
Sy = ParIdentity(T, gy)
Sc = ParMatrix(T, gc, gc)
```

Chain our operators and distribute them:

```julia
S = Sc ⊗ Sy ⊗ Sx
S = distribute(S, partition)
```

Parametrize and apply our transform:

```julia
θ = init(S)
S(θ) * vec(x)
```

Take the gradient of the parameters w.r.t to some objective by simply doing:

```julia
θ′ = gradient(θ -> sum(S(θ) * vec(x)), θ)
```

Another out-of-box example can be seen at [Distributed Parametrized Convolution of a 3D Tensor](@ref)

## Sharing Weights

Sharing weights can be thought of as a broadcasting operation.

In order to share weights of an operator across multiple processing elements, we can do:

```julia
A = ParMatrix(T, 20, 20)
A = distribute(A)
```

Assume the following partition and tensor shape:

```julia
gc, gx = 20, 100
partition = [1, 4]

nc, nx = [gc, gx] .÷ partition
x = rand(T, nc, nx)
```

Initialize and apply the matrix operator on the sharded tensor:

```julia
θ = init(A)
A(θ) * x
```

Compute the gradient by doing:

```julia
θ′ = gradient(θ -> sum(A(θ) * x), θ)
```

## Reduction Operation

In order to perform a reduction operation, more commonly known as an `ALL_REDUCE` operation, we can define:

```julia
R = ParReduce(T)
```

Given any local vector or matrix, we can do:

```julia
x = rand(T, 100)
R * x
```

To compute the gradient of the input w.r.t some objective:

```julia
x′ = gradient(x -> sum(R * x), x)
```
