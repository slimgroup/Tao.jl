### Distributed FFT of a 3D Tensor

Make sure to add necessary dependencies. You might also need to load a proper MPI implementation based on your hardware.

```julia
julia> using Pkg
julia> Pkg.activate("path/to/your/environment")
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

# Julia requires you to manually assign the gpus, modify to your case.
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
