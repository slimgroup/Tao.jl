### Distributed FFT of a 3D Tensor

!!! note "Jump right in"
    To get started, you can run some [examples](https://github.com/turquoisedragon2926/ParametricOperators-Examples)

Make sure to add necessary dependencies. You might also need to load a proper MPI implementation based on your hardware.

```julia
julia> ]
(v1.9) activate /path/to/your/environment
(env) add MPI CUDA ParametricOperators
```

!!! warning "To run on multiple GPUs"
    If you wish to run on multiple GPUs and the below code fails to scale, make sure the GPUs are binded to different tasks. The approach we use is to unbind our GPUs on request and assign manually:

    ```julia
    CUDA.device!(rank % 4)
    ```

    which might be different if you have more or less than 4 GPUs per node. Also, make sure your MPI distribution is functional.
```julia
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

If you have [`mpiexecjl`](https://juliaparallel.org/MPI.jl/stable/usage/#Installation) set up, you can run the above by doing:

```shell
mpiexecjl --project=/path/to/your/environment -n NTASKS julia code_above.jl
```

OR if you have a HPC cluster with [`slurm`](https://slurm.schedmd.com/documentation.html) set up, you can do:

```shell
salloc --gpus=NTASKS --time=01:00:00 --ntasks=NTASKS --gpus-per-task=1 --gpu-bind=none
srun julia --project=/path/to/your/environment code_above.jl
```

!!! warning "Allocation"
    Your `salloc` might look different based on your HPC cluster
