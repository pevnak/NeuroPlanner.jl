
"""
find_duplicates(o)

create a `mask` identifying unique columns in `o` and a map `si`
mapping indexes original columns in `o` to the new representation in `o[:,mask]`
It should hold that `o[:,mask][:,[si[i] for i in 1:size(o,2)]] == o`

The implementation is based on hashing columns
"""
find_duplicates(x::AbstractMatrix) = find_duplicates(vec(mapslices(hash, x, dims = 1)))

function find_duplicates(xs::AbstractVector...)
	h = hash.(first(xs))
	for x in Base.tail(xs)
		h .= hash.(x,h)
	end
	find_duplicates(h)
end


function find_duplicates(x::Vector{T}) where {T<:Number}
	cols = length(x)
	si = Vector{Int}(undef, cols)
	unique_cols = Vector{Bool}(undef, cols)
	di = Dict{T, Int}()
	@inbounds for j in 1:cols 
		n = length(di)
		si[j] = get!(di, x[j], n + 1)
		unique_cols[j] = n < length(di)
	end
	unique_cols, si
end

# Custom key with overwritten hash function allows to bypass the hashing, since when values are already hashed
# https://discourse.julialang.org/t/dictionary-with-custom-hash-function/49168
struct HashedKey
    val::UInt64
end

Base.hash(a::HashedKey, h::UInt) = xor(a.val, h)
Base.:(==)(a::HashedKey, b::HashedKey) = a.val == b.val

function find_duplicates(x::Vector{UInt64})
	n = length(x)
	si = Vector{Int}(undef, n)
	uniques = Vector{Bool}(undef, n)
	di = Dict{HashedKey, Int}()
	@inbounds for j in 1:n 
		u = length(di)
		k = HashedKey(x[j])
		si[j] = get!(di, k, u + 1)
		uniques[j] = u < length(di)
	end
	uniques, si
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

MLUtils.numobs(ds::DeduplicatingNode) = length(ds.ii)
Base.getindex(x::DeduplicatingNode, ii) = DeduplicatingNode(x.x, x.ii[ii])
Base.getindex(x::DeduplicatingNode, ii::Vector{Bool}) = DeduplicatingNode(x.x, x.ii[ii])
Base.getindex(x::DeduplicatingNode, ii::Vector{Int}) = DeduplicatingNode(x.x, x.ii[ii])
HierarchicalUtils.children(n::DeduplicatingNode) = (n.x,)
function HierarchicalUtils.nodecommshow(io::IO, n::DeduplicatingNode)
	bytes = Base.format_bytes(Base.summarysize(n.ii))
    print(io, " # ", length(n.ii), " obs, ", bytes)
end

(m::AbstractMillModel)(dedu::DeduplicatingNode{<:AbstractMillNode}) = DeduplicatedMatrix(m(dedu.x), dedu.ii)
(m::AbstractMillModel)(kb::KnowledgeBase, dedu::DeduplicatingNode{<:AbstractMillNode}) =  DeduplicatedMatrix(m(kb, dedu.x), dedu.ii)
# (m::AbstractMillModel)(dedu::DeduplicatingNode{<:AbstractMillNode}) = m(dedu.x)[:,dedu.ii]
# (m::AbstractMillModel)(kb::KnowledgeBase, dedu::DeduplicatingNode{<:AbstractMillNode}) = m(kb, dedu.x)[:,dedu.ii]

function deduplicate(kb::KnowledgeBase)
	ii = Int[]
	for k in keys(kb)
		new_entry, ii = _deduplicate(kb, kb[k])
		kb = replace(kb, k, new_entry)
	end
	# if length(ii) == length(unique(ii))
	# 	println("all entries are unique")
	# else 
	# 	p = round(100*(1 - length(unique(ii)) / length(ii)), digits = 2)
	# 	println("$(p) entries are duplicate")
	# end
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

_sethash(xs::AbstractVector) = sum(hash.(xs)) # this might be good enough for the bang
# _sethash(xs::AbstractVector) = hash(sort(xs)) # this probably more correct

function _deduplicate(kb::KnowledgeBase, ds::BagNode)
	if numobs(ds.data) == 0 
		@assert all(isempty(b) for b in ds.bags) "All bags should be empty when instances are empty"
		dedu_bn = BagNode(ds.data, [0:-1])
		ii = ones(Int, numobs(ds))
		dedu_ds = DeduplicatingNode(dedu_bn, ii)
		return(dedu_ds, ii)
	end
	z, ii = _deduplicate(kb, ds.data)
	mapped_bags = [ii[b] for b in ds.bags]
	mask, ii = find_duplicates(_sethash.(mapped_bags))
	dedu_bn = z isa DeduplicatingNode ? BagNode(z.x, mapped_bags[mask]) : BagNode(z, ds.bags[mask])
	dedu_ds = DeduplicatingNode(dedu_bn, ii)
	dedu_ds, ii
end

function _deduplicate(kb::KnowledgeBase, ds::ProductNode)
	xs = _map_tuple(Base.Fix1(_deduplicate, kb), ds.data)
	mask, ii = find_duplicates(_map_tuple(Base.Fix2(getindex,2), xs)...)
	dedu_ds = ProductNode(map(x -> x[1][mask], xs))
	dedu_ds = DeduplicatingNode(dedu_ds, ii)
	dedu_ds, ii
end


function _deduplicate(kb::KnowledgeBase, ds::MaskedNode)
    x, ii = _deduplicate(kb, ds.data)
    mask, ii = find_duplicates(ii, ds.mask)
    dedu_ds = MaskedNode(x[mask], ds.mask[mask])
    dedu_ds = DeduplicatingNode(dedu_ds, ii)
    dedu_ds, ii
end


