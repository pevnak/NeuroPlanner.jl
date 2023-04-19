
"""
find_duplicates(o)

create a `mask` identifying unique columns in `o` and a map `si`
mapping indexes original columns in `o` to the new representation in `o[:,mask]`
It should hold that `o[:,mask][:,[si[i] for i in 1:size(o,2)]] == o`
"""
function find_duplicates(x::AbstractMatrix)
	rows, cols = size(x)
	si = Vector{Int}(undef, cols)
	unique_cols = Vector{Bool}(undef, cols)
	new_index = 1
	@inbounds for j in 1:cols 
		si[j] = new_index
		unique_cols[j] = true
		for k in 1:(j-1)
			!unique_cols[k] && continue
			if all(x[r,k] == x[r,j] for r in 1:rows)
				si[j] = si[k]
				unique_cols[j] = false
				break
			end
		end
		new_index += unique_cols[j]
	end
	unique_cols, si
end

function find_duplicates(x::AbstractVector)
	cols = length(x)
	si = Vector{Int}(undef, cols)
	unique_cols = Vector{Bool}(undef, cols)
	new_index = 1
	@inbounds for j in 1:cols 
		si[j] = new_index
		unique_cols[j] = true
		for k in 1:(j-1)
			!unique_cols[k] && continue
			if x[k] == x[j]
				si[j] = si[k]
				unique_cols[j] = false
				break
			end
		end
		new_index += unique_cols[j]
	end
	unique_cols, si
end

"""
struct DeduplicatingNode{D} 
	x::D
	i::Vector{Int}
end
A deduplicated data are useful for speeding computation of models, which returns 
the same answer over multiple observations. The idea is that the result is computed as
m(ddd::DeduplicatingNode) =  m(ddd.x)[:,ddd.colmap], where we assume that m(ddd.x) returns 
a matrix.
"""
struct DeduplicatingNode{D} <: AbstractMillNode
	x::D
	colmap::Vector{Int}
end

Mill.nobs(ds::DeduplicatingNode) = length(ds.colmap)
Base.getindex(x::DeduplicatingNode, ii) = DeduplicatingNode(x.x, x.colmap[ii])
Base.getindex(x::DeduplicatingNode, ii::Vector{Bool}) = DeduplicatingNode(x.x, x.colmap[ii])
Base.getindex(x::DeduplicatingNode, ii::Vector{Int}) = DeduplicatingNode(x.x, x.colmap[ii])
HierarchicalUtils.children(n::DeduplicatingNode) = (n.x,)
function HierarchicalUtils.nodecommshow(io::IO, n::DeduplicatingNode)
	bytes = Base.format_bytes(Base.summarysize(n.colmap))
    print(io, " # ", length(n.colmap), " obs, ", bytes)
end

function deduplicate_data(model::AbstractMillModel, ds::AbstractMillNode)
	deduplicate_data(model(ds), ds)
end

function deduplicate_data(kb::KnowledgeBase, model::AbstractMillModel, ds::AbstractMillNode)
	deduplicate_data(model(kb, ds), ds)
end

function deduplicate_data(output::AbstractMatrix, ds::AbstractMillNode)
	mask, si = find_duplicates(output)
	DeduplicatingNode(ds[mask], [si[i] for i in 1:Mill.nobs(ds)])
end

deduplicate_data(model, ds) = ds

(m::AbstractMillModel)(dedu::DeduplicatingNode{<:AbstractMillNode}) = m(dedu.x)[:,dedu.colmap]
(m::AbstractMillModel)(kb::KnowledgeBase, dedu::DeduplicatingNode{<:AbstractMillNode}) = m(kb, dedu.x)[:,dedu.colmap]


"""
deduplicate(model::BagModel, ds::BagNode)

remove duplicates in instances of BagNodes for 
ds = BagNode(ArrayNode([1 2 1 2 2; 2 1 2 1 2]), [1:3,4:5])
model = reflectinmodel(ds, d -> Dense(d,32), SegmentedMean) 
dds = deduplicate(model, ds)
dds.data.data ≈ [1 2 2; 2 1 2]
dds.bags.bags ≈ [[1, 2, 1], [2, 3]]

"""
deduplicate(kb, model::ArrayModel, ds::ArrayNode) = deduplicate_data(kb, model, ds)

@generated function deduplicate(kb::KnowledgeBase, model::ProductModel{<:NamedTuple{KM}}, ds::ProductNode{<:NamedTuple{KM}}) where {KM}
    chs = map(KM) do k
        :(deduplicate_data(kb, model.ms.$k, ds.data.$k))
    end
    quote
        ProductNode(NamedTuple{KM}(tuple($(chs...))))
    end
end

function deduplicate(kb::KnowledgeBase, model::ProductModel{<:Tuple}, ds::ProductNode{<:Tuple})
    chs = [deduplicate_data(kb, m, x) for (m,x) in zip(model.ms, ds.data)]
    ProductNode((tuple(chs...)))
end

function deduplicate(kb::KnowledgeBase, model::BagModel, ds::BagNode)
	subds = deduplicate_data(kb, model.im, ds.data)
	BagNode(subds, ds.bags)
end

function deduplicate(model::KnowledgeModel, kb::KnowledgeBase)
	# Let's first completely evaluate the knowledgebase
	kb = _apply_layers(kb, model)
	xs = map(keys(kb)) do k 
		k ∉ keys(model) && return(kb[k])
		dedu = deduplicate(kb, model[k], kb[k])
		deduplicate_data(kb[k], dedu)
	end 
	KnowledgeBase(NamedTuple{keys(kb)}(xs))
end



function _deduplicate(kb::KnowledgeBase)
	for k in keys(kb)
		kb = replace(kb, k, _deduplicate(kb, kb[k])[1])
	end
	kb
end

function _deduplicate(kb::KnowledgeBase, x::AbstractMatrix) 
	o = DeduplicatedMatrix(x)
	o, o.ii
end

# function _deduplicate(kb::KnowledgeBase, ds::KBEntry{K}) where {K}
# 	x = kb[K]
# 	@assert x isa DeduplicatedMatrix # this is absolutely dirty
# 	ii = find_duplicates(x.ii[ds.ii])[2]
# 	o = KBEntry{K}(ii)
# 	o, o.ii
# end


function _deduplicate(kb::KnowledgeBase, ds::ArrayNode{<:KBEntry{K}}) where {K}
	x = kb[K]
	mask, ii = x isa DeduplicatedMatrix ? find_duplicates(x.ii[ds.data.ii]) : find_duplicates(x)
	dedu_ds = DeduplicatingNode(ds[mask], ii)
	dedu_ds, ii
end

function _deduplicate(kb::KnowledgeBase, ds::BagNode)
	z, ii = _deduplicate(kb, ds.data)
	mapped_bags = [sort(ii[b]) for b in ds.bags]
	mask, ii = find_duplicates(mapped_bags)
	dedu_bn = z isa DeduplicatingNode ? BagNode(z.x, mapped_bags[mask]) : BagNode(z, ds.bags[mask])
	dedu_ds = DeduplicatingNode(dedu_bn, ii)
	dedu_ds, ii
end

function _deduplicate(kb::KnowledgeBase, ds::ProductNode)
	xs = map(x -> _deduplicate(kb, x), ds.data)
	mask, ii = find_duplicates(vcat(map(x -> x[2]', xs)...))
	dedu_ds = ProductNode(map(x -> x[1][mask], xs))
	dedu_ds = DeduplicatingNode(dedu_ds, ii)
	dedu_ds, ii
end


