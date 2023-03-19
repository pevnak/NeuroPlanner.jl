"""
struct RSearchTree{T}
	st::Dict{UInt64,Vector{T}}
end

Is a reversed search tree, where every node (key) contains list of 
its childs. RSearchTree simplifies construction of minibatches from 
a solution.
"""
struct RSearchTree{T}
	st::Dict{UInt64,Set{T}}
end

function RSearchTree(st::Dict{UInt64, T}) where {T<:SymbolicPlanners.PathNode{GenericState}}
	rst = Dict{UInt64,Set{T}}()
	for node in values(st)	
		parent_id = node.parent_id
		parent_id === nothing && continue
		v = get!(rst, parent_id, Set{T}())
		push!(v, node)
	end
	RSearchTree(rst)
end