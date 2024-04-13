using ParametricOperators
using CUDA
using MPI
using Zygote

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
