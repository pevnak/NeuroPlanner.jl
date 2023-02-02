"""
leafs(search_tree)

identify all leafs (states without parents) in  `search_tree`

"""
function leafs(search_tree::Dict)
	l = Set(keys(search_tree))
	for v in values(search_tree)
		v.parent_id ∈ l && pop!(l, v.parent_id)
	end
	l
end

"""
artificial_trajectory(sol)

samples randomly leaf from a search tree and returns the trajectory

"""
function artificial_trajectory(sol)
	isempty(sol.search_tree) && error("cannot create an artificial goal from empty search tree")
	search_tree = sol.search_tree
	id = rand(leafs(search_tree))
	trajectory =  Vector{GenericState}()
	while(true)
		v = search_tree[id]
		push!(trajectory, v.state)
		v.parent_id === nothing && break
		id = v.parent_id
	end
	reverse(trajectory)
end
