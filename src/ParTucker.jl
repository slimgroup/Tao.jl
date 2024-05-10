export ParTucker

"""
Dense N dim Tensor in compressed Tucker format
"""


struct ParTucker{T,N}  <: ParLinearOperator{T,T,Parametric,External}
    core::ParMatrix{T} # Actually ParTensor core G reshaped to G^(1)
    factors::Vector
    input_dimension::Vector #only for [nx ny nt]
    restriction::Vector #[mx my mt]
    block_diagonals::Vector #temporary add to test
    id::Any
    ParTucker(c::ParMatrix,f::Vector,i::Vector,restr::Vector,block_diagonals::Vector) = new{DDT(c),length(f)}(c,f,i,restr,block_diagonals,uuid4(Random.GLOBAL_RNG))
    ParTucker(c::ParMatrix,f::Vector,i::Vector,restr::Vector,block_diagonals::Vector,id::Any) = new{DDT(c),length(f)}(c,f,i,restr,block_diagonals,id)
end

# (A::ParTucker)(layer::Int) = ParTucker(A.core,[A.factors[1:end-1]..., A.factors[end][:,layer]],A.input_dimension,A.restriction)
# function ParTucker(type::DataType,size::Vector,rank::Vector,restriction::Vector)
#     factors = []
#     n = length(size)
    
#     core = ParMatrix(type,rank[1],prod(rank[2:n]))
#     push!(factors,ParMatrix(type,size[1],rank[1]))
#     for i  = 2:length(size)
#         push!(factors,ParMatrix(type,rank[i],size[i]))
#     end
#     return ParTucker(core,factors,restriction)
# end

# function ParTucker(type::DataType,size::Vector,rank::Vector,restriction::Vector,id::Any)
#     factors = []
#     n = length(size)
    
#     core = ParMatrix(type,rank[1],prod(rank[2:n]))
#     push!(factors,ParMatrix(type,size[1],rank[1]))
#     for i  = 2:length(size)
#         push!(factors,ParMatrix(type,rank[i],size[i]))
#     end
#     return ParTucker(core,factors,restriction,id)
# end


function TuckerShape(A::ParTucker)
    shape = []
    push!(shape,Range(A.factors[1]))

    for j = 2:length(A.factors)
        push!(shape,Domain(A.factors[j]))
    end
    return shape
end

function TuckerRank(A::ParTucker)
    rank = []
    push!(rank,Domain(A.factors[1]))
    for j = 2:length(A.factors)
        push!(rank,Range(A.factors[j]))
    end
    return rank
end
#
# (A::ParTucker{D,R,L,External})(params) where {D,R,L} = ParParameterized(A, params[A])

function Domain(A::ParTucker)
    i = Domain(A.factors[2])
    restr_x = 2*A.restriction[1]
    restr_y = 2*A.restriction[2]
    restr_t = A.restriction[3]
    return i*restr_x*restr_y*restr_t
end

function Range(A::ParTucker)
    o = Range(A.factors[1])
    restr_x = 2*A.restriction[1]
    restr_y = 2*A.restriction[2]
    restr_t = A.restriction[3]
    return o*restr_x*restr_y*restr_t
end

function init!(A::ParTucker{T,N}, d::Parameters)where {N,T}
     d[A] = rand(T,2,2) # key - value pair needed for ParTucker
    init!(A.core,d)
    for j = 1:length(A.factors)
        init!(A.factors[j],d)
    end
    for j = 1:length(A.block_diagonals)
        init!(A.block_diagonals[j],d)
    end
end

# # Kronecker multiplication for spectral convolution for 2D FNO 
# function (w::ParParameterized{T,T,Linear,ParTucker{T,6},V})(x::X) where {T,V,X<:AbstractMatrix{T}}
#     b = size(x,2)
#     o = Range(w.op.factors[1]) # U1 is o \times k_1
#     i = Domain(w.op.factors[2]) # U2 is k_2 \times i
#     mx = w.op.restriction[1]; my = w.op.restriction[2]; mt = w.op.restriction[3]
#     x = reshape(x,(i,b,2*mx,2*my,mt))
#     z = x
#     G = reshape(w.op.core.params,TuckerRank(w.op))
#     # y = ein"abcdef,ia,jb,kc,ld,me,nf,jpklm->ipklm"(G,w.op.factors[1],)
#     y = ein"abcdef,ia,bj,ck,dl,em,fn,jpklm->ipklm"(G,w.op.factors[1],w.op.factors[2:end]...,z)
#     # y = reshape(y,(o,b,2*mx,2*my,mt))  
#     y = reshape(y,(:,b))
#     return y
#  end


function to_Dict(A::ParTucker{T,N}) where {N,T}

    rv = Dict{String, Any}(
        "type" => "ParTucker",
        "T" => string(T),
        "shape" => TuckerShape(A),
        "rank" => TuckerRank(A),
        "restriction" => A.restriction
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

function from_Dict(::Type{ParTucker{T,N}}, d) where {N,T}
    ts = d["T"]
    if !haskey(Data_TYPES, ts)
        throw(ParException("unknown data type `$ts`"))
    end
    dtype = Data_TYPES[ts]
    mid = d["id"]
    if startswith(mid, "UUID:")
        mid = UUID(mid[6:end])
    end
    ParTucker(dtype, d["shape"], d["rank"], d["restriction"], mid)
end

#### some scrap from past code 

#  function (w::ParParameterized{T,T,Linear,ParTucker{T,6},V})(x::X) where {T,V,X<:AbstractMatrix{T}}
#     b = size(x,2)
#     o = Range(w.op.factors[1]) # U1 is o \times k_1
#     i = Domain(w.op.factors[2]) # U2 is k_2 \times i
#     # nx = size(x,3)
#     # ny = size(x,4)
#     # ny = size(x,5)
#     Id = ParIdentity(T,b)
#     mx = w.op.restriction[1]; my = w.op.restriction[2]; mt = w.op.restriction[3]
#     # x = reshape(x,(i,:))
#     # x = reshape(x,(:,b))
#     # x = restrict_dft(x)
#     # #given x as prod * batch ---> reshape to i,b,2*mx,2*my,mt
#     x = reshape(x,(i,b,2*mx,2*my,mt))
#     # z = x


 
    # y = vcat([(Id ⊗ (w.op.factors[1].params*w.op.core.params*(w.op.factors[6].params⊗w.op.factors[5][:,k].params⊗w.op.factors[4][:,j].params ⊗
    # w.op.factors[3][:,i].params⊗w.op.factors[2].params)))*
    #   vec(z[:,:,i,j,k]) for i = 1:Domain(w.op.factors[3]), j = 1:Domain(w.op.factors[4]), k = 1:Domain(w.op.factors[5])]...)
      
    # y = vcat([(Id ⊗ (w.factors[1](theta)*w.core(theta)*(w.factors[6](theta)⊗w.factors[5](theta)[:,k]⊗w.factors[4](theta)[:,j] ⊗
    #   w.factors[3](theta)[:,i]⊗w.factors[2](theta))))*
    #     vec(z[:,:,i,j,k]) for i = 1:Domain(w.factors[3]), j = 1:Domain(w.factors[4]), k = 1:Domain(w.factors[5])]...)