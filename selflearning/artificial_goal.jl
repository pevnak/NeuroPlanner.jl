using Dictionaries

"""
find_leafs(sol)

identify all states without successors (leafs) in search tree `sol.search_tree`

"""
function find_leafs(sol)
	search_tree = sol.search_tree
	st = map(i -> true, Indices(keys(search_tree)))
	for v in values(search_tree)
		v.parent_id === nothing && continue
		st[v.parent_id] = false
	end
	keys(filter(identity, st))
end

"""
artificial_trajector(sol)

samples randomly leaf from a search tree and returns the trajectory

"""
function artificial_trajectory(sol)
	isempty(sol.search_tree) && error("cannot create an artificial goal from empty search tree")
	search_tree = sol.search_tree
	id = rand(find_leafs(sol))
	trajectory =  Vector{GenericState}()
	while(true)
		v = search_tree[id]
		push!(trajectory, v.state)
		v.parent_id === nothing && break
		id = v.parent_id
	end
	reverse(trajectory)
end
