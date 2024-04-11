using Pkg
Pkg.activate("./")

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

# Global and Local Partition
gt, gx, gy = 100, 100, 100
nt, nx, ny = [gt, gx, gy] .÷ partition

Ft = ParDFT(T, gt)
Fx = ParDFT(Complex{T}, gx)
Fy = ParDFT(Complex{T}, gy)

F = Fy ⊗ Fx ⊗ Ft
F = distribute(F, partition)

x = rand(T, nt, nx, ny) |> gpu
y = F * vec(x)

MPI.Finalize()
