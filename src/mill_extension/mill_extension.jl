module MillExtension

using Mill
import Mill: catobs

using Flux
using HierarchicalUtils
using MLUtils
using HierarchicalUtils

import Base: *, ==

# COMMON ALIASES
using Base: AbstractVecOrMat, AbstractVecOrTuple
const VecOrRange{T} = Union{UnitRange{T},AbstractVector{T}}
const VecOrTupOrNTup{T} = Union{Vector{<:T},Tuple{Vararg{T}},NamedTuple{K,<:Tuple{Vararg{T}}} where K}
const Maybe{T} = Union{T,Missing}
const Optional{T} = Union{T,Nothing}

const DOCTEST_FILTER = r"\s*-?[0-9]+\.[0-9]+[\.]*\s*"

const AbstractMillStruct = Union{AbstractMillModel,AbstractMillNode}

include("datanodes/datanode.jl")
export AbstractMaskedNode
export MaskedNode

include("modelnodes/modelnode.jl")
export MaskedModel

include("show.jl")

include("hierarchical_utils.jl")

end
