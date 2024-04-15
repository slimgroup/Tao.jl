### FFT of 3D Tensor

!!! note "Jump right in"
    To get started, you can run some [examples](https://github.com/turquoisedragon2926/ParametricOperators.jl-Examples)

```julia
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

Run the above by doing:
```shell
julia --project=/path/to/your/environment code_above.jl
```
