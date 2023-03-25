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
find_duplicates(o)

create a `mask` identifying unique columns in `o` and a map `si`
mapping indexes original columns in `o` to the new representation in `o[:,mask]`
It should hold that `o[:,mask][:,[si[i] for i in 1:size(o,2)]] == o`
"""
function find_duplicates(o)
	ii = duplicated_columns(o)
	mask = falses(size(o, 2))
	si = zeros(Int, size(o, 2))
	for (i, k) in enumerate(ii)
		if !haskey(index_map, k)
			index_map[k] = length(index_map) + 1
			mask[i] = true
		end 
		si[i] = index_map[k]
	end
	mask, si
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

deduplicate(kb, model::ArrayModel, ds::ArrayNode) = ds

function deduplicate(kb, model::ArrayModel, ds::ArrayNode{<:KBEntry})

end

@generated function deduplicate(kb::KnowledgeBase, model::ProductModel{<:NamedTuple{KM}}, ds::ProductNode{<:NamedTuple{KM}}) where {KM}
    chs = map(KM) do k
        :(deduplicate(kb, model.ms.$k, ds.data.$k))
    end
    quote
        ProductNode(NamedTuple{KM}(tuple($(chs...))))
    end
end

function deduplicate(kb::KnowledgeBase, model::ProductModel{<:Tuple}, ds::ProductNode{<:Tuple})
    chs = [deduplicate(kb, m, x) for (m,x) in zip(model.ms, ds.data)]
    ProductNode((tuple(chs...)))
end


function deduplicate(kb::KnowledgeBase, model::BagModel, ds::BagNode)
	subds = deduplicate(kb, model.im, ds.data)
	mask, si = find_duplicates(model.im(kb, subds))
	BagNode(
		ds.data[mask],
		ScatteredBags(map(b -> si[b], ds.bags)),
	)
end

function deduplicate(model::KnowledgeModel, ds::KnowledgeBase)
	# Let's first completely evaluate the knowledgebase
	kb = _apply_layers(ds, model)
	xs = map(keys(ds)) do k 
		k ∉ keys(model) && return(ds[k])
		deduplicate(kb, model[k], ds[k])
	end 
	KnowledgeBase(NamedTuple{keys(ds)}(xs))
end
