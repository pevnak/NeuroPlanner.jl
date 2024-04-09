using DataStructures: SortedDict, OrderedDict

"""
    CompressedBags{T <: Integer} <: AbstractBags{T}

[`CompressedBags`](@ref) struct stores indices of bags' instances that are not necessarily contiguous.

See also: [`AlignedBags`](@ref).
"""
struct CompressedBags{T<:Integer} <: AbstractBags{T}
    indices::Vector{T}
    ranges::Vector{UnitRange{Int64}}
    n::Int64
    function CompressedBags(indices::Vector{T}, ranges::Vector{UnitRange{Int64}}, n::Int64) where {T}
        @assert length(indices) == sum(length.(ranges)) "Dimensionality mismatch, number of observations in `indices` must match the `ranges`.
        `number of observations in `indices` = $(length(indices)) and ranges size = $(sum(length.(ranges)))."
        new{T}(indices, ranges, n)
    end
end

Flux.@forward CompressedBags.indices Base.getindex, Base.setindex!, Base.firstindex, Base.lastindex,
Base.eachindex, Base.first, Base.last, Base.iterate, Base.eltype, Base.length


MLUtils.numobs(b::CompressedBags) = b.n
# Base.length(b::CompressedBags) = length(b.ranges)

Base.mapreduce(f, op, b::CompressedBags) = mapreduce(f, op, b.indices)
maxindex(b::CompressedBags) = isempty(b) ? -1 : b.n



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

The function calculates the starting points of each bag based on the cumulative sum of `counts` and then creates ranges for each bag using `map`. Finally, it initializes `vals` as a vector of uninitialized integers of length `n`.
"""
function CompressedBags(ks::Vector{T}, counts::Vector{Int}, n::Int) where {T<:Integer}
    ends = cumsum(counts)
    start = ends .- (counts .- 1)
    ranges = map((x, y) -> x:y, start, ends)
    indices = Vector{Int}(undef, length(ks))

    for (i, k) in enumerate(ks)
        indices[start[k]] = i % n == 0 ? n : i % n
        start[k] += 1
    end

    CompressedBags(indices, ranges, n)
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
    ranges = Vector{UnitRange{Int}}(undef, length(idcs))
    indices = Vector{T}(undef, sum(length.(b.ranges[idcs])))
    m = OrderedDict{T,Int}((v => i for (i, v) in enumerate(unique(vcat(b.indices[vcat(b.ranges[idcs]...)]...)))))

    offset = 0
    for (i, j) in enumerate(idcs)
        start = offset + 1
        for ii in b.ranges[j]
            offset += 1
            indices[offset] = m[b.indices[ii]]
        end
        ranges[i] = start:offset
    end

    CompressedBags(indices, ranges, length(keys(m))), collect(keys(m))
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
    n = sum(mask)
    ranges = Vector{UnitRange{Int}}(undef, length(b.ranges))
    indices = Vector{T}()
    offset = 0
    for (i, r) in enumerate(b.ranges)
        start = offset + 1
        for ii in r
            mask[b.indices[ii]] || continue
            offset += 1
            push!(indices, m[b.indices[ii]])
        end
        ranges[i] = start:offset
    end

    CompressedBags(indices, ranges, n)
end



function _catbags(bs::Vector{CompressedBags{T}}) where {T<:Integer}
    indices = Vector{T}(undef, sum(length.(bs)))
    ranges = Vector{UnitRange{Int64}}(undef, sum([length(b.ranges) for b in bs]))

    n = 0
    offsetᵢ = 0
    offsetᵣ = 0

    for b in bs
        for (i, idx) in enumerate(b.indices)
            indices[offsetᵢ+i] = idx + n
        end

        for (i, r) in enumerate(b.ranges)
            ranges[offsetᵣ+i] = r .+ offsetᵢ
        end

        offsetᵢ += length(b.indices)
        offsetᵣ += length(b.ranges)
        n += b.n
    end

    CompressedBags(indices, ranges, n)
end



"""
Base.enumerate(b::CompressedBags)
"""
function Base.enumerate(b::CompressedBags)
    return Base.Iterators.enumerate(@view b.indices[range] for range in b.ranges)
end


Base.hash(e::CompressedBags, h::UInt) = hash((e.indices, e.ranges, e.n), h)
e1::CompressedBags == e2::CompressedBags = e1.indices == e2.indices && e1.ranges == e2.ranges && e1.n == e2.n

