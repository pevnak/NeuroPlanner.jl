function CompressedBagNode(d::T, b::CompressedBags, m=nothing) where {T<:Maybe{AbstractMillNode}}
    @assert(numobs(d) == b.num_observations,
        "The number of observations in data is not the same as in the bag. `numobs(data)` = $(numobs(d)) and `num_observations` = $(b.num_observations)")
    BagNode(d, b, m)
end
