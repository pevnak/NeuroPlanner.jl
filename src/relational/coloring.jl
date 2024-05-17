"""
	function color(kb::KnowledgeBase)
	
	Compute color of each vertex in the  knowledge base. This is used mainly to extract features
	from the graph for the Weisfeler-leman graph kernel. The color of a vertex is a hash of the
	vertex and its neighbors. The color is computed through the knowledge base, which means that
	if you want multipled message passes, they have to be present in the KnowledgeBase.
"""
function color(kb::KnowledgeBase{KS,VS}) where {KS,VS}
	o = KnowledgeBase(NamedTuple())
	color(o, kb)
end


function color(o::KnowledgeBase, kb::KnowledgeBase{KS,VS}) where {KS,VS}
	isempty(kb) && return(o)
	k = first(KS)
	o = append(o, k, color(o, first(kb.kb)))
	color(o, Base.tail(kb))
end

color(x::AbstractMatrix) = vec(mapslices(hash, x, dims = 1))

color(kb::KnowledgeBase, x::AbstractMatrix) = color(x)

function color(kb::KnowledgeBase, ds::ArrayNode{<:KBEntry})
	K = ds.data.e
	x = kb[K]
	(x isa DeduplicatedMatrix || x isa DeduplicatingNode) && error("I do not color DeduplicatingNode")
	x[ds.data.ii]
end

function color(kb::KnowledgeBase, ds::BagNode)
	if numobs(ds.data) == 0 
		@assert all(isempty(b) for b in ds.bags) "All bags should be empty when instances are empty"
		return(fill(hash(UInt64[]), length(ds.bags)))
	end
	x = color(kb, ds.data)
	map(b -> _sethash(x[b]), ds.bags)
end

function color(kb::KnowledgeBase, ds::ProductNode)
	xs = _map_tuple(Base.Fix1(color, kb), ds.data)
	o = map(hash, zip(xs...))
	o
end

function color(kb::KnowledgeBase, ds::MaskedNode)
    x = fill(0x77cfa1eef01bca90, length(ds.mask))
    ds[mask] .= color(kb, ds.data)
    x
end


