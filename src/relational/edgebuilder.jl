mutable struct EdgeBuilder{N,I<:Integer}
	ii::NTuple{N,Vector{I}}
	bags::Vector{<:Vector{I}}
	nv::Integer
	first_free::I 
	function EdgeBuilder(ii::NTuple{N,Vector{I}}, bags::Vector{<:Vector{I}}, nv, first_free::I) where {N, I<:Integer}
		N > 0 || error("Can create only edge of positive arity.")
		first_free == 1 || error("EdgeBuilder should be initialized with empty counters")
		n = length(first(ii))
		all(length(i) == n for i in ii) || error("length of all vectors to hold vertex indices should be the same")
		length(bags) == nv || error("length of bags should be equal to the length of vertices")
		new{N,I}(ii, bags, nv, first_free)
	end
end

EdgeBuilder(arity::Integer, capacity::Integer, nv::Integer) = EdgeBuilder(Int, arity, capacity, nv)

function EdgeBuilder(I::DataType, arity::Integer, capacity::Integer, nv::Integer)
	ii = _map_tuple(i -> Vector{I}(undef, capacity), Val{arity})
	bags = [I[] for _ in 1:nv]
	EdgeBuilder(ii, bags, nv, 1)
end

function Base.push!(eb::EdgeBuilder{N,<:Any}, vertices::NTuple{N,I}) where {N,I<:Integer}
	Base.@boundscheck boundscheck(eb, vertices)
	_mapenumerate_tuple(vertices) do i, vᵢ
		push!(eb.bags[vᵢ], eb.first_free)
		eb.ii[i][eb.first_free] = vᵢ
	end
	eb.first_free += 1
end

function boundscheck(eb::EdgeBuilder, vertices)
	eb.first_free ≤ length(first(eb.ii)) || error("The capacity of edgebuilder has exceeded")
	all(_map_tuple(≤(eb.nv), vertices)) || error("index of vertices cannot be bigger then number of vertices")
	true	
end

function construct(eb::EdgeBuilder, input_sym::Symbol)
	l = eb.first_free -1 
	xs = _map_tuple(ii -> KBEntry(input_sym, view(ii, 1:l)), eb.ii)
	BagNode(ProductNode(xs), eb.bags)
end
