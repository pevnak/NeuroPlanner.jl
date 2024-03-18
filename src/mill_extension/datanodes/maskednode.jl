"""
    MaskedNode{A <: AbstractMillNode, M<:BitVector} <: AbstractMillNode

    Data node for storing any Mill node data of type `A` and a bit mask for all observations in Mill node data of type BitVector.
    
# Examples
```jldoctest masked_node
julia> n = ProductNode(a=ArrayNode([0 1; 2 3]), b=ArrayNode([4 5; 6 7]))
ProductNode  # 2 obs, 16 bytes
    ├── a: ArrayNode(2×2 Array with Int64 elements)  # 2 obs, 80 bytes
    ╰── b: ArrayNode(2×2 Array with Int64 elements)  # 2 obs, 80 bytes

julia> mn = MaskedNode(n, BitVector([1, 0]))
MaskedNode  # 2 obs, 96 bytes
    ├── ProductNode  # 2 obs, 16 bytes
    │     ├── a: ArrayNode(2×2 Array with Int64 elements)  # 2 obs, 80 bytes
    │     ╰── b: ArrayNode(2×2 Array with Int64 elements)  # 2 obs, 80 bytes
    ╰── Bool[1, 0]
```

See also: [`AbstractMillNode`](@ref), [`MaskedModel`](@ref), [`ProductNode`](@ref).
"""
struct MaskedNode{A<:AbstractMillNode,M<:BitVector} <: AbstractMaskedNode
    data::A
    mask::M
    function MaskedNode(data::A, mask::M) where {A,M}
        @assert numobs(data) == length(mask) "Dimensionality mismatch, number of samples in `data` needs to be equal to the length of the `mask`.
        'mask length = $(length(mask)) and number of samples = $(numobs(data))"
        new{A,M}(data, mask)
    end
end

"""
    MaskedNode(d<:AbstractMillNode)

Construct a new [`MaskedNode`](@ref) with data `d` and a full mask.

```jldoctest masked_model
julia> Random.seed!(0);

julia> n = ProductNode(x1 = ArrayNode(Flux.onehotbatch([1, 2], 1:2)),
                       x2 = BagNode(ArrayNode([1 2; 3 4]), [1:2, 0:-1]),
                       x3 = ArrayNode(rand(2, 2)))
ProductNode  # 2 obs, 40 bytes
├── x1: ArrayNode(2×2 OneHotArray with Bool elements)  # 2 obs, 80 bytes
├── x2: BagNode  # 2 obs, 96 bytes
│         ╰── ArrayNode(2×2 Array with Int64 elements)  # 2 obs, 80 bytes
╰── x3: ArrayNode(2×2 Array with Float64 elements)  # 2 obs, 80 bytes

julia> mn = MaskedNode(n)
MaskedNode  # 2 obs, 120 bytes
  ├── ProductNode  # 2 obs, 40 bytes
  │     ├── x1: ArrayNode(2×2 OneHotArray with Bool elements)  # 2 obs, 80 bytes
  │     ├── x2: BagNode  # 2 obs, 96 bytes
  │     │         ╰── ArrayNode(2×2 Array with Int64 elements)  # 2 obs, 80 bytes
  │     ╰── x3: ArrayNode(2×2 Array with Float64 elements)  # 2 obs, 80 bytes
  ╰── Bool[1, 1]
```
See also: [`AbstractMillNode`](@ref), [`MaskedModel`](@ref), [`ProductNode`](@ref).
"""
MaskedNode(d::AbstractMillNode) = MaskedNode(d, BitVector(ones(Bool, numobs(d))))

Flux.@functor MaskedNode

mapdata(f, x::MaskedNode) = MaskedNode(mapdata(f, x.data), x.mask)

dropmeta(x::MaskedNode) = MaskedNode(dropmeta(x.data), x.mask)

Base.size(x::MaskedNode) = size(x.data)

function Base.reduce(::typeof(catobs), as::Vector{<:MaskedNode})
    MaskedNode(reduce(catobs, data.(as)), reduce(vcat, [x.mask for x in as]))
end

Base.hash(e::MaskedNode, h::UInt) = hash((e.data, e.mask), h)
(e1::MaskedNode == e2::MaskedNode) = isequal(e1.data == e2.data, true) && e1.mask == e2.mask
Base.isequal(e1::MaskedNode, e2::MaskedNode) = isequal(e1.data, e2.data) && isequal(e1.mask, e2.mask)





