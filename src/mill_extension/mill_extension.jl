import Mill: catobs

import Base: *, ==

# COMMON ALIASES
using Base: AbstractVecOrMat, AbstractVecOrTuple
const VecOrRange{T} = Union{UnitRange{T},AbstractVector{T}}
const VecOrTupOrNTup{T} = Union{Vector{<:T},Tuple{Vararg{T}},NamedTuple{K,<:Tuple{Vararg{T}}} where K}
const Maybe{T} = Union{T,Missing}
const Optional{T} = Union{T,Nothing}

include("maskednode.jl")
export AbstractMaskedNode
export MaskedNode

include("maskedmodel.jl")
export MaskedModel


import Mill: _levelparams, _show_submodels
_levelparams(m::MaskedModel) = Flux.params(m.m)


import HierarchicalUtils: NodeType, LeafNode, children
@nospecialize
NodeType(::Type{<:BitVector}) = LeafNode()
children(n::MaskedNode) = (n.data, n.mask)
children(n::MaskedModel) = (n.m,)
@specialize
