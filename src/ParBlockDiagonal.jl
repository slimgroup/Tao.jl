export ParBlockDiagonal, ⊠

struct ParBlockDiagonal{D,R,P,F,N} <: ParLinearOperator{D,R,P,Internal}
    ops::F
    function ParBlockDiagonal(ops...)
        ops = collect(ops)
        N = length(ops)

        # We can't BD only a single operator
        if N == 1
            return ops[1]
        end

        DDTs = map(DDT, ops)
        RDTs = map(RDT, ops)

        @assert all(el -> typeof(el) == typeof(DDTs[1]), DDTs)
        @assert all(el -> typeof(el) == typeof(RDTs[1]), RDTs)

        D = DDTs[1]
        R = RDTs[1]
        P = foldl(promote_parametricity, map(parametricity, ops))
        return new{D,R,P,typeof(ops),N}(ops)
    end
    function ParBlockDiagonal(D::DataType,R::DataType,P,ops)
        return new{D,R,P,typeof(ops),length(ops)}(ops)
    end
end

⊠(A::ParLinearOperator, B::ParLinearOperator) = ParBlockDiagonal(A, B)
⊠(A::ParBlockDiagonal, B::ParLinearOperator) = ParBlockDiagonal([A.ops..., B]...)
⊠(A::ParLinearOperator, B::ParBlockDiagonal) = ParBlockDiagonal([A, B.ops...]...)
⊠(A::ParBlockDiagonal, B::ParBlockDiagonal) = ParBlockDiagonal([A.ops..., B.ops...]...)

Domain(A::ParBlockDiagonal) = sum(map(Domain, children(A)))
Range(A::ParBlockDiagonal) = sum(map(Range, children(A)))

children(A::ParBlockDiagonal) = A.ops
rebuild(::ParBlockDiagonal, cs) = ParBlockDiagonal(cs...)

adjoint(A::ParBlockDiagonal{D,R,P,F,N}) where {D,R,P,F,N} = ParBlockDiagonal(reverse(collect(map(adjoint, A.ops)))...)

function (A::ParBlockDiagonal{D,R,<:Applicable,F,N})(x::X) where {D,R,F,N,X<:AbstractMatrix{D}}
    start_idx = 1
    results = []

    for (i, op) in enumerate(A.ops)
        end_idx = start_idx + Domain(op) - 1

        push!(results, op(x[start_idx:end_idx, :]))
        start_idx = end_idx + 1
    end

    return vcat(results...)
end

(A::ParBlockDiagonal{D,R,<:Applicable,F,N})(x::X) where {D,R,F,N,X<:AbstractVector{D}} = vec(A(reshape(x, length(x), 1)))

function ChainRulesCore.rrule(A::ParBlockDiagonal{D,R,<:Applicable,F,N}, x::X) where {D,R,F,N,X<:AbstractMatrix{D}}
    op_out = A(x)
    function pullback(op)
        return (NoTangent(), A'(op))
    end
    return op_out, pullback
end
