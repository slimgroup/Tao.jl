using ParametricOperators
using CUDA
using MPI

MPI.Init()

comm = MPI.COMM_WORLD
rank = MPI.Comm_rank(comm)
size = MPI.Comm_size(comm)

CUDA.device!(rank % 4)
partition = [1, size]

T = Float32

nx, ny, nz = 10, 10, 10

Ft = ParDFT(T, nt)
Fx = ParDFT(Complex{T}, nx)
Fy = ParDFT(Complex{T}, ny)

F = Fx ⊗ Ft
F = distribute(F, partition)

x = rand(T, nt, nx) |> gpu
y = F * x

MPI.Finalize()
