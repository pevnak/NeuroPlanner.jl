
"""
    AbstractMaskedNode <: AbstractMillNode

    
Supertype for any data node structure representing a multi-instance learning problem with mask.
"""
abstract type AbstractMaskedNode <: AbstractMillNode end

"""
    Mill.data(n::AbstractMillNode)

Return data stored in node `n`.

# Examples
```jldoctest
julia> Mill.data(ArrayNode([1 2; 3 4], "metadata"))
2×2 Matrix{Int64}:
 1  2
 3  4

julia> Mill.data(BagNode(ArrayNode([1 2; 3 4]), [1, 2], "metadata"))
2×2 ArrayNode{Matrix{Int64}, Nothing}:
 1  2
 3  4
```

See also: [`Mill.metadata`](@ref)
"""
data(n::AbstractMillNode) = n.data

include("maskednode.jl")
Base.ndims(::MaskedNode) = Colon()
MLUtils.numobs(x::MaskedNode) = numobs(x.data)

catobs(as::Maybe{MaskedNode}...) = reduce(catobs, collect(as))

