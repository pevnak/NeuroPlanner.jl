using DataStructures: SortedDict, OrderedDict

"""
    CompressedBags{T <: Integer} <: AbstractBags{T}

[`CompressedBags`](@ref) struct stores indices of bags' instances that are not necessarily contiguous.

See also: [`AlignedBags`](@ref).
"""
struct CompressedBags{T<:Integer} <: AbstractBags{T}
    indices::Vector{T}
    bags::Vector{UnitRange{Int64}}
    num_observations::Int64
    function CompressedBags(indices::Vector{T}, bags::Vector{UnitRange{Int64}}, num_observations::Int64) where {T}
        @assert length(indices) == sum(length.(bags)) "Dimensionality mismatch, number of observations in `indices` must match the `bags`.
        `number of observations in `indices` = $(length(indices)) and bags size = $(sum(length.(bags)))."
        new{T}(indices, bags, num_observations)
    end
end

Flux.@forward CompressedBags.bags Base.firstindex, Base.lastindex,
Base.eachindex, Base.eltype

Base.getindex(b::CompressedBags, i::Int) = view(b.indices, b.bags[i])
Base.getindex(b::CompressedBags, I::AbstractUnitRange{<:Integer}) = [b[i] for i in I]

Base.first(b::CompressedBags) = b[1]
function Base.first(b::CompressedBags, n::Int)
    n < 0 && throw(ArgumentError("Number of elements must be nonnegative"))
    n > length(b) && return b[1:end]
    b[1:n]
end

Base.last(b::CompressedBags) = b[end]
function Base.last(b::CompressedBags, n::Int)
    n < 0 && throw(ArgumentError("Number of elements must be nonnegative"))
    n > length(b) && return b[1:end]
    b[end-1:end]
end



function Base.iterate(bags::CompressedBags, i=1)
    i > length(bags.bags) && return (nothing)
    bags[i], i + 1
end

MLUtils.numobs(b::CompressedBags) = length(b.bags)
Base.length(b::CompressedBags) = length(b.bags)

Base.mapreduce(f, op, b::CompressedBags) = mapreduce(f, op, b.indices)
maxindex(b::CompressedBags) = isempty(b) ? -1 : b.num_observations





"""
Base.enumerate(b::CompressedBags)
"""
# function Base.enumerate(b::CompressedBags)
# return Base.Iterators.enumerate(@view b.indices[range] for range in b.bags)
# end



"""
    CompressedBags()

Construct a new [`CompressedBags`](@ref) struct containing no bags.

# Examples
```jldoctest
julia> CompressedBags()
CompressedBags{Int64}(Int64[], UnitRange{Int64}[], 0)
```
"""
CompressedBags() = CompressedBags(Vector{Int}(), Vector{UnitRange{Int}}(), 0)

"""
    CompressedBags(k::Vector{<:Integer})

Construct a new [`CompressedBags`](@ref) struct from `Vector` `k` specifying the index of the bag each instance belongs to.


TODO 

# Examples
```jldoctest
julia> CompressedBags([1, 1, 5, 2, 3, 4, 2, 3], [2, 2, 2, 1, 1], 4)
CompressedBags{Int64}([1, 2, 4, 3, 1, 4, 2, 3], UnitRange{Int64}[1:2, 3:4, 5:6, 7:7, 8:8], 4)
```
"""


"""
    CompressedBags(k::Vector{T}, counts::Vector{Int}, n::Int) where {T<:Integer}

Create a vector `vals` representing compressed bags based on the input vectors `k` and `counts`.

# Arguments
- `k::Vector{T}`: Vector of integers representing the ids of objects in bags.
- `counts::Vector{Int}`: Vector of integers representing the counts of elements in each bag.
- `n::Int`: Total number of observations.

# Returns
- `vals::Vector{Int}`: Vector representing the compressed bags.

# Description
This function takes two input vectors `k` and `counts`, where `k` represents the distinct elements in the bags and `counts` represents the counts of elements in each bag. It creates a vector `vals` representing the compressed bags, where each element of `vals` corresponds to the count of elements from `k` in the bags.

The function calculates the starting points of each bag based on the cumulative sum of `counts` and then creates bags for each bag using `map`. Finally, it initializes `vals` as a vector of uninitialized integers of length `n`.
"""
function CompressedBags(ks::Vector{T}, counts::Vector{Int}, num_observations::Int) where {T<:Integer}
    ends = cumsum(counts)
    start = ends .- (counts .- 1)
    bags = map((x, y) -> x:y, start, ends)
    indices = Vector{Int}(undef, length(ks))

    for (i, k) in enumerate(ks)
        # indices[start[k]] = i % num_observations == 0 ? num_observations : i % num_observations
        indices[start[k]] = (i - 1) % num_observations + 1
        start[k] += 1
    end

    CompressedBags(indices, bags, num_observations)
end

"""
    remapbags(b::AbstractBags, idcs::VecOrRange{<:Integer}) -> (rb, I)

Select a subset of bags in `b` corresponding to indices `idcs` and remap instance indices appropriately.
Return new bags `rb` as well as a `Vector` of remapped instances `I`.

# Examples
```jldoctest
julia> remapbags(CompressedBags([1, 2, 4, 3, 1, 4, 2, 3], [1:2, 3:4, 5:6, 7:7, 8:8], 4), [1, 3])
(CompressedBags{Int64}([1, 2, 1, 3], UnitRange{Int64}[1:2, 3:4], 3), [1, 2, 4])
```
"""

function remapbags(b::CompressedBags{T}, idcs::VecOrRange{<:Int}) where {T}
    bags = Vector{UnitRange{Int}}(undef, length(idcs))
    indices = Vector{T}(undef, sum(length.(b.bags[idcs])))
    m = OrderedDict{T,Int}((v => i for (i, v) in enumerate(unique(vcat(b.indices[vcat(b.bags[idcs]...)]...)))))

    offset = 0
    for (i, j) in enumerate(idcs)
        start = offset + 1
        for ii in b.bags[j]
            offset += 1
            indices[offset] = m[b.indices[ii]]
        end
        bags[i] = start:offset
    end

    CompressedBags(indices, bags, length(keys(m))), collect(keys(m))
end


"""
    adjustbags(b::CompressedBags, mask::AbstractVector{Bool})

Remove indices of instances brom bags `b` and remap the remaining instances accordingly.

# Examples
```jldoctest
julia> adjustbags(CompressedBags([1, 2, 3, 4, 1, 4, 2, 3], [1:2, 3:4, 5:6, 7:7, 8:8], 4), [false, false, true, true])
CompressedBags{Int64}([1, 2, 2, 1], UnitRange{Int64}[1:0, 1:2, 3:3, 4:3, 4:4], 2)
```
"""
function adjustbags(b::CompressedBags{T}, mask::AbstractVecOrMat{Bool}) where {T}
    m = cumsum(mask)
    bags = Vector{UnitRange{Int}}(undef, length(b.bags))
    indices = Vector{T}()
    offset = 0
    for (i, r) in enumerate(b.bags)
        start = offset + 1
        for ii in r
            mask[b.indices[ii]] || continue
            offset += 1
            push!(indices, m[b.indices[ii]])
        end
        bags[i] = start:offset
    end

    CompressedBags(indices, bags, sum(mask))
end



function _catbags(bs::Vector{CompressedBags{T}}) where {T<:Integer}
    indices = Vector{T}(undef, sum([length(b.indices) for b in bs]))
    bags = Vector{UnitRange{Int64}}(undef, sum(length.(bs)))

    num_observations = 0
    offsetᵢ = 0
    offsetᵣ = 0

    for b in bs
        for (i, idx) in enumerate(b.indices)
            indices[offsetᵢ+i] = idx + num_observations
        end

        for (i, r) in enumerate(b.bags)
            bags[offsetᵣ+i] = r .+ offsetᵢ
        end

        offsetᵢ += length(b.indices)
        offsetᵣ += length(b.bags)
        num_observations += b.num_observations
    end

    CompressedBags(indices, bags, num_observations)
end


Base.hash(e::CompressedBags, h::UInt) = hash((e.indices, e.bags, e.num_observations), h)
e1::CompressedBags == e2::CompressedBags = e1.indices == e2.indices && e1.bags == e2.bags && e1.num_observations == e2.num_observations

