# ParametricOperators.jl

[![][license-img]][license-status] 
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
   julia> Pkg.activate("path/to/your/project")
   julia> Pkg.add("ParametricOperators")
   ```

This will add `ParametricOperators.jl` as dependency to your project

## Examples

### 1. FFT of 3D Tensor

```julia
using Pkg
Pkg.activate("./path/to/your/environment")

using ParametricOperators

T = Float32

gt, gx, gy = 100, 100, 100

# Define a transform along each dimension
Ft = ParDFT(T, gt)
Fx = ParDFT(Complex{T}, gx)
Fy = ParDFT(Complex{T}, gy)

# Create a Kronecker operator than chains together the transforms
F = Fy ⊗ Fx ⊗ Ft

# Apply the transform on a random input
x = rand(T, gt, gx, gy) |> gpu
y = F * vec(x)
```

### 2. Distributed FFT of a 3D Tensor:

Make sure to add necessary dependencies. You might also need to load a proper MPI implementation based on your hardware.

```julia
julia> using Pkg
julia> Pkg.activate("path/to/your/project")
julia> Pkg.add("MPI")
julia> Pkg.add("CUDA")
```

Copy the following code into a `.jl` file
```julia
using Pkg
Pkg.activate("./path/to/your/environment")

using ParametricOperators
using CUDA
using MPI

MPI.Init()

comm = MPI.COMM_WORLD
rank = MPI.Comm_rank(comm)
size = MPI.Comm_size(comm)

CUDA.device!(rank % 4)
partition = [1, 1, size]

T = Float32

# Define your Global Size and Data Partition
gt, gx, gy = 100, 100, 100
nt, nx, ny = [gt, gx, gy] .÷ partition

# Define a transform along each dimension
Ft = ParDFT(T, gt)
Fx = ParDFT(Complex{T}, gx)
Fy = ParDFT(Complex{T}, gy)

# Create and distribute the Kronecker operator than chains together the transforms
F = Fy ⊗ Fx ⊗ Ft
F = distribute(F, partition)

# Apply the transform on a random input
x = rand(T, nt, nx, ny) |> gpu
y = F * vec(x)

MPI.Finalize()
```

You can run the above by doing:

`srun -n N_TASKS julia code_above.jl`

### 3. Parametrized Convolution on 3D Tensor

Make sure to add necessary dependencies to compute the gradient

```julia
julia> using Pkg
julia> Pkg.activate("path/to/your/project")
julia> Pkg.add("Zygote")
```

```julia
using Pkg
Pkg.activate("./path/to/your/environment")

using ParametricOperators
using Zygote

T = Float32

gt, gx, gy = 100, 100, 100

# Define a transform along each dimension
St = ParMatrix(T, gt, gt)
Sx = ParMatrix(T, gx, gx)
Sy = ParMatrix(T, gy, gy)

# Create a Kronecker operator than chains together the transforms
S = Sy ⊗ Sx ⊗ St

# Parametrize our transform
θ = init(S) |> gpu

# Apply the transform on a random input
x = rand(T, gt, gx, gy) |> gpu
y = S(θ) * vec(x)

# Compute the gradient wrt some objective of our parameters
θ′ = gradient(θ -> sum(S(θ) * vec(x)), θ)
```

### 4. Distributed Parametrized Convolution of a 3D Tensor:

Make sure to add necessary dependencies. You might also need to load a proper MPI implementation based on your hardware.

```julia
julia> using Pkg
julia> Pkg.activate("path/to/your/project")
julia> Pkg.add("MPI")
julia> Pkg.add("CUDA")
julia> Pkg.add("Zygote")
```

Copy the following code into a `.jl` file
```julia
using Pkg
Pkg.activate("./path/to/your/environment")

using ParametricOperators
using CUDA
using MPI

MPI.Init()

comm = MPI.COMM_WORLD
rank = MPI.Comm_rank(comm)
size = MPI.Comm_size(comm)

CUDA.device!(rank % 4)
partition = [1, 1, size]

T = Float32

# Define your Global Size and Data Partition
gt, gx, gy = 100, 100, 100
nt, nx, ny = [gt, gx, gy] .÷ partition

# Define a transform along each dimension
St = ParMatrix(T, gt, gt)
Sx = ParMatrix(T, gx, gx)
Sy = ParMatrix(T, gy, gy)

# Create and distribute the Kronecker operator than chains together the transforms
S = Sy ⊗ Sx ⊗ St
S = distribute(S, partition)

# Parametrize our transform
θ = init(S) |> gpu

# Apply the transform on a random input
x = rand(T, nt, nx, ny) |> gpu
y = S(θ) * vec(x)

# Compute the gradient wrt some objective of our parameters
θ′ = gradient(θ -> sum(S(θ) * vec(x)), θ)

MPI.Finalize()
```

You can run the above by doing:

`srun -n N_TASKS julia code_above.jl`
<!-- ## Citation

If you use our software for your research, we appreciate it if you cite us following the bibtex in [CITATION.bib](CITATION.bib). -->

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
