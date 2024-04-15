mutable struct EdgeBuilderComp{T<:Integer}
    indices::Vector{T}
    counts::Vector{T}
    arity::Int
    num_observations::Int
    offset::Int
    function EdgeBuilderComp(indices::Vector{T}, counts::Vector{T}, arity::Int, num_observations::Int, offset::Int) where {T}
        arity > 0 || error("Can create only edge of positive arity.")
        offset == 0 || error("Offset needs to start at 0.")
        length(indices) == arity * num_observations || error("Length of indices must be equal to `arity * num_observations`.")

        new{T}(indices, counts, arity, num_observations, offset)
    end
end

function EdgeBuilderComp(::Val{arity}, num_observations::Int, numobs::Int) where {arity}
    indices = Vector{Int}(undef, arity * num_observations)
    counts = fill(0, numobs)
    offset = 0
    EdgeBuilderComp(indices, counts, arity, num_observations, offset)
end


function EdgeBuilderComp(arity::Int, num_observations::Int, numobs::Int)
    indices = Vector{Int}(undef, arity * num_observations)
    counts = fill(0, numobs)
    offset = 0
    EdgeBuilderComp(indices, counts, arity, num_observations, offset)
end

function Base.push!(eb::EdgeBuilderComp, vertices::NTuple{N,I}) where {N,I<:Integer}
    eb.offset += 1
    _mapenumerate_tuple(vertices) do i, vᵢ
        eb.counts[vᵢ] += 1
        eb.indices[(i-1)*eb.num_observations+eb.offset] = vᵢ
    end
end

function construct(eb::EdgeBuilderComp, input_sym::Symbol)
    xs = Tuple([ArrayNode(KBEntry(input_sym, eb.indices[(i-1)*eb.num_observations+1:(i-1)*eb.num_observations+eb.offset])) for i in 1:eb.arity])
    CompressedBagNode(ProductNode(xs), CompressedBags(eb.indices, eb.counts, eb.num_observations, eb.offset))
end
