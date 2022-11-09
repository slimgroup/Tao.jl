export DDT, RDT, linearity, parametricity, order
export Domain, Range, children, nparams, init

DDT(::ParOperator{D,R,L,P,O})           where {D,R,L,P,O} = D
RDT(::ParOperator{D,R,L,P,O})           where {D,R,L,P,O} = R
linearity(::ParOperator{D,R,L,P,O})     where {D,R,L,P,O} = L
parametricity(::ParOperator{D,R,L,P,O}) where {D,R,L,P,O} = P
order(::ParOperator{D,R,L,P,O})         where {D,R,L,P,O} = O

promote_linearity(::Type{Linear}, ::Type{Linear}) = Linear
promote_linearity(::Type{<:Linearity}, ::Type{<:Linearity}) = NonLinear

promote_parametricity(::Type{NonParametric}, ::Type{NonParametric}) = NonParametric
promote_parametricity(::Type{<:Applicable}, ::Type{<:Applicable}) = Parameterized
promote_parametricity(::Type{Parametric}, ::Type{<:Applicable}) = Parametric
promote_parametricity(::Type{<:Applicable}, ::Type{Parametric}) = Parametric
promote_parametricity(::Type{Parametric}, ::Type{Parametric}) = Parametric

promote_opdim(::Nothing, d) = d
promote_opdim(d, ::Nothing) = d
promote_opdim(::Nothing, ::Nothing) = nothing
promote_opdim(d1, d2) = d1 == d2 ? d1 : throw(ParException("Incompatible operator dimensions in ParAdd: $((d1, d2))"))

function Domain end
function Range end
function children end
function nparams end
function init end

children(::ParOperator{D,R,L,P,FirstOrder}) where {D,R,L,P} = nothing

nparams(A::ParOperator{D,R,L,P,O}) where {D,R,L,P<:Applicable,O} = 0
nparams(A::ParOperator{D,R,L,Parametric,HigherOrder}) where {D,R,L} = mapreduce(nparams, sum, children(A))

init(::ParOperator{D,R,L,P,O}) where {D,R,L,P<:Applicable,O} = nothing
init(A::ParOperator{D,R,L,Parametric,HigherOrder}) where {D,R,L} = optional_multitype_vcat(map(init, children(A))...)

(A::ParOperator{D,R,L,P,O})(x::X) where {D,R,L,P<:Applicable,O,X<:AbstractMatrix{D}} = mapreduce(A, hcat, eachcol(x))
(A::ParOperator{Nothing,R,L,P,O})(_) where {R,L,P<:Applicable,O} = A()

*(A::ParLinearOperator{D,R,P,O}, x::X) where {D,R,P<:Applicable,O,X<:AbstractVector{D}} = A(x)
*(A::ParLinearOperator{D,R,P,O}, x::X) where {D,R,P<:Applicable,O,X<:AbstractMatrix{D}} = A(x)
*(x::X, A::ParLinearOperator{D,R,P,O}) where {D,R,P<:Applicable,O,X<:AbstractMatrix{R}} = (A'*x')'