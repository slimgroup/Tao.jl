### Parametrized Convolution on 3D Tensor

Make sure to add necessary dependencies to compute the gradient

```julia
julia> using Pkg
julia> Pkg.activate("path/to/your/environment")
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