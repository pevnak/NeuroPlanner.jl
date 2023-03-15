"""
duplicated_columns(x::Matrix{T}) where {T<:Number}

find duplicated columns in matrix x

```julia
julia> duplicated_columns([1 2 1 2 2; 2 1 2 1 2])
5-element Vector{Int64}:
 1
 2
 1
 2
 5
```

"""
function duplicated_columns(x::Matrix{T}) where {T<:Number}
	rows, cols = size(x)
	ii = collect(1:cols)
	for j in 2:cols 
		for k in 1:j
			if all(x[r,k] == x[r,j] for r in 1:rows)
				ii[j] = k
				break
			end
		end
	end
	ii
end


"""
deduplicate(model::BagModel, ds::BagNode)

remove duplicates in instances of BagNodes for 
ds = BagNode(ArrayNode([1 2 1 2 2; 2 1 2 1 2]), [1:3,4:5])
model = reflectinmodel(ds, d -> Dense(d,32), SegmentedMean) 
dds = deduplicate(model, ds)
dds.data.data ≈ [1 2 2; 2 1 2]
dds.bags.bags ≈ [[1, 2, 1], [2, 3]]

"""

deduplicate(model::ArrayModel, ds::ArrayNode) = ds

@generated function deduplicate(model::ProductModel{<:NamedTuple{KM}}, ds::ProductNode{<:NamedTuple{KM}}) where {KM}
    chs = map(KM) do k
        :(deduplicate(model.ms.$k, ds.data.$k))
    end
    quote
        ProductNode(NamedTuple{KM}(tuple($(chs...))))
    end
end


function deduplicate(model::BagModel, ds::BagNode)
	subds = deduplicate(model.im, ds.data)
	o = model.im(subds)
	mask = falses(size(o, 2))
	index_map = Dict{Int,Int}()
	ii = duplicated_columns(o)
	si = zeros(Int, size(o, 2))
	for (i, k) in enumerate(ii)
		if !haskey(index_map, k)
			index_map[k] = length(index_map) + 1
			mask[i] = true
		end 
		si[i] = index_map[k]
	end
	BagNode(
		ds.data[mask],
		ScatteredBags(map(b -> si[b], ds.bags)),
	)
end