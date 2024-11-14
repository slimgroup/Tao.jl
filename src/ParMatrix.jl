export ParMatrix

"""
Dense matrix operator.
"""
struct ParMatrix{T} <: ParLinearOperator{T,T,Parametric,External}
    m::Int
    n::Int
    id::Any
    init::Function

    function ParMatrix{T}(m::Int, n::Int, id::Any=uuid4(Random.GLOBAL_RNG), init::Function=rand) where T
        new{T}(m, n, id, init)
    end

    ParMatrix(T::DataType, m::Int, n::Int, id::Any=uuid4(Random.GLOBAL_RNG), init::Function=rand)=ParMatrix{T}(m, n, id, init)
    ParMatrix(T::DataType, m::Int, n::Int, init::Function) = ParMatrix{T}(m, n, uuid4(Random.GLOBAL_RNG), init)
    ParMatrix(m::Int, n::Int, id::Any=uuid4(Random.GLOBAL_RNG), init::Function=rand)=ParMatrix{Float64}(m, n, id, init)
    ParMatrix(m::Int, n::Int, init::Function) = ParMatrix{Float64}(m, n, uuid4(Random.GLOBAL_RNG), init)
end

Domain(A::ParMatrix) = A.n
Range(A::ParMatrix) = A.m

complexity(A::ParMatrix{T}) where {T} = elementwise_multiplication_cost(T)*A.n*A.m

distribute(A::ParMatrix) = ParBroadcasted(A)

# function init!(A::ParMatrix{T}, d::Parameters) where {T}
#     d[A] = A.init(T, A.m, A.n)
# end

function init!(A::ParMatrix{T}, d::Parameters) where {T<:Real}
    println("OLD INIT")
    if A.n == 1
        d[A] = rand(T, A.m, A.n) # TODO: Make init function passable
        return
    end
    scale = sqrt(24.0f0 / sum((A.m, A.n)))
    d[A] = (rand(T, (A.n, A.m)) .- 0.5f0) .* scale
    d[A] = permutedims(d[A], [2, 1])
end

function init!(A::ParMatrix{T}, d::Parameters) where {T<:Complex}
    if A.n == 1
        d[A] = rand(T, A.m, A.n) # TODO: Make init function passable
        return
    end
    d[A] = rand(T, A.n, A.m)/convert(real(T), sqrt(A.m*A.n))
    d[A] = permutedims(d[A], [2, 1])
end

(A::ParParameterized{T,T,Linear,ParMatrix{T},V})(x::X) where {T,V,X<:AbstractVector{T}} = A.params*x
(A::ParParameterized{T,T,Linear,ParMatrix{T},V})(x::X) where {T,V,X<:AbstractMatrix{T}} = A.params*x
(A::ParParameterized{T,T,Linear,ParMatrix{T},V})(x::X) where {T,V,X<:AbstractArray{T,3}} = batched_mul(A.params,x)
(A::ParParameterized{T,T,Linear,ParAdjoint{T,T,Parametric,ParMatrix{T}},V})(x::X) where {T,V,X<:AbstractVector{T}} = A.params'*x
(A::ParParameterized{T,T,Linear,ParAdjoint{T,T,Parametric,ParMatrix{T}},V})(x::X) where {T,V,X<:AbstractMatrix{T}} = A.params'*x

*(x::X, A::ParParameterized{T,T,Linear,ParMatrix{T},V}) where {T,V,X<:AbstractVector{T}} = x*A.params
*(x::X, A::ParParameterized{T,T,Linear,ParMatrix{T},V}) where {T,V,X<:AbstractMatrix{T}} = x*A.params
*(x::X, A::ParParameterized{T,T,Linear,ParAdjoint{T,T,Parametric,ParMatrix{T}},V}) where {T,V,X<:AbstractVector{T}} = x*A.params'
*(x::X, A::ParParameterized{T,T,Linear,ParAdjoint{T,T,Parametric,ParMatrix{T}},V}) where {T,V,X<:AbstractMatrix{T}} = x*A.params'

+(x::X, A::ParParameterized{T,T,Linear,ParMatrix{T},V}) where {T,V,X<:AbstractVector{T}} = x.+A.params
+(x::X, A::ParParameterized{T,T,Linear,ParMatrix{T},V}) where {T,V,X<:AbstractArray{T}} = x.+A.params
+(x::X, A::ParParameterized{T,T,Linear,ParMatrix{T},V}) where {T,V,X<:AbstractMatrix{T}} = x.+A.params
+(x::X, A::ParParameterized{T,T,Linear,ParAdjoint{T,T,Parametric,ParMatrix{T}},V}) where {T,V,X<:AbstractVector{T}} = x+A.params'
+(x::X, A::ParParameterized{T,T,Linear,ParAdjoint{T,T,Parametric,ParMatrix{T}},V}) where {T,V,X<:AbstractArray{T}} = x+A.params'
+(x::X, A::ParParameterized{T,T,Linear,ParAdjoint{T,T,Parametric,ParMatrix{T}},V}) where {T,V,X<:AbstractMatrix{T}} = x+A.params'

/(A::ParParameterized{T,T,Linear,ParMatrix{T},V}, x::X) where {T,V,X<:AbstractMatrix{T}} = A.params./x

function to_Dict(A::ParMatrix{T}) where {T}
    rv = Dict{String, Any}(
        "type" => "ParMatrix",
        "T" => string(T),
        "m" => A.m,
        "n" => A.n
    )
    if typeof(A.id) == String
        rv["id"] = A.id
    elseif typeof(A.id) == UUID
        rv["id"] = "UUID:$(string(A.id))"
    else
        throw(ParException("I don't know how to encode id of type $(typeof(A.id))"))
    end
    rv
end

function from_Dict(::Type{ParMatrix}, d)
    ts = d["T"]
    if !haskey(Data_TYPES, ts)
        throw(ParException("unknown data type `$ts`"))
    end
    dtype = Data_TYPES[ts]
    mid = d["id"]
    if startswith(mid, "UUID:")
        mid = UUID(mid[6:end])
    end
    ParMatrix(dtype, d["m"], d["n"], mid)
end

function Base.getindex(A::ParMatrix{T}, rows, cols) where T
    row_range = isa(rows, Colon) ? (1:Range(A)) : (isa(rows, Integer) ? (rows:rows) : rows)
    col_range = isa(cols, Colon) ? (1:Domain(A)) : (isa(rows, Integer) ? (cols:cols) : cols)

    new_m = length(row_range)
    new_n = length(col_range)

    return ParMatrix(T, new_m, new_n, A.init) # TODO: Track IDs?
end

function Base.getindex(A::ParParameterized{T,T,Linear,ParMatrix{T},V}, rows, cols) where {T,V}
    row_range = isa(rows, Colon) ? (1:Range(A)) : (isa(rows, Integer) ? (rows:rows) : rows)
    col_range = isa(cols, Colon) ? (1:Domain(A)) : (isa(rows, Integer) ? (cols:cols) : cols)

    new_m = length(row_range)
    new_n = length(col_range)
    
    new_params = reshape(A.params[rows, cols], new_m, new_n)
    new_matrix = ParMatrix(T, new_m, new_n, A.op.init) # TODO: Track IDs?

    return ParParameterized(new_matrix, new_params)
end
