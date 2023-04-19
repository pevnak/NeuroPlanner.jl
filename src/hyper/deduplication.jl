
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
	ii::Vector{Int}
end
A deduplicated data are useful for speeding computation of models, which returns 
the same answer over multiple observations. The idea is that the result is computed as
m(ddd::DeduplicatingNode) =  m(ddd.x)[:,ddd.ii], where we assume that m(ddd.x) returns 
a matrix.
"""
struct DeduplicatingNode{D} <: AbstractMillNode
	x::D
	ii::Vector{Int}
end

Mill.nobs(ds::DeduplicatingNode) = length(ds.ii)
Base.getindex(x::DeduplicatingNode, ii) = DeduplicatingNode(x.x, x.ii[ii])
Base.getindex(x::DeduplicatingNode, ii::Vector{Bool}) = DeduplicatingNode(x.x, x.ii[ii])
Base.getindex(x::DeduplicatingNode, ii::Vector{Int}) = DeduplicatingNode(x.x, x.ii[ii])
HierarchicalUtils.children(n::DeduplicatingNode) = (n.x,)
function HierarchicalUtils.nodecommshow(io::IO, n::DeduplicatingNode)
	bytes = Base.format_bytes(Base.summarysize(n.ii))
    print(io, " # ", length(n.ii), " obs, ", bytes)
end

(m::AbstractMillModel)(dedu::DeduplicatingNode{<:AbstractMillNode}) = m(dedu.x)[:,dedu.ii]
(m::AbstractMillModel)(kb::KnowledgeBase, dedu::DeduplicatingNode{<:AbstractMillNode}) = m(kb, dedu.x)[:,dedu.ii]

function deduplicate(kb::KnowledgeBase)
	for k in keys(kb)
		kb = replace(kb, k, _deduplicate(kb, kb[k])[1])
	end
	kb
end

function _deduplicate(kb::KnowledgeBase, x::AbstractMatrix) 
	o = DeduplicatedMatrix(x)
	o, o.ii
end

function _deduplicate(kb::KnowledgeBase, ds::ArrayNode{<:KBEntry{K}}) where {K}
	x = kb[K]
	mask, ii = (x isa DeduplicatedMatrix || x isa DeduplicatingNode) ? find_duplicates(x.ii[ds.data.ii]) : find_duplicates(x)
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


