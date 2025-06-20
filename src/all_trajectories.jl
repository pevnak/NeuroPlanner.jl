
"""
	is_root(k, st)

	true if node `k` is root in search_tree `st`
"""
is_root(k::UInt64, v::SymbolicPlanners.PathNode) = v.parent.id == k
is_root(k::UInt64, st::Dict{UInt64, SymbolicPlanners.PathNode{GenericState}}) = is_root(k, st[k])

"""
	add_all_parents(search_tree, trajectories)

	Extend all trajectories by parents unless 
	if the trajectory is not complete (the first state is root)
"""
function add_all_parents(search_tree, trajectories::Vector{Vector{UInt64}})
	new_trajectries = Vector{Vector{UInt64}}()
	for trajectory in trajectories
		k = trajectory[1]
		v = search_tree[k]
		parent = v.parent
		if is_root(k, search_tree)  # skip items where we have reached the head
			push!(new_trajectries, trajectory)
			continue
		end

		while parent !== nothing
			if search_tree[parent.id].path_cost < v.path_cost  # this is to avoid cycles
				push!(new_trajectries, vcat([parent.id], trajectory))
			end
			parent = parent.next
		end
	end
	new_trajectries
end

"""
	all_trajectories_id(search_tree, trajectories)

	Extend all trajectories by parents in the search_tree. 
	If the trajectory is not complete (the first state is root)
"""
function all_trajectories_id(search_tree, goal_state; subset_fun = maxstates_subset, kwargs...)
	all_trajectories_id(search_tree, goal_state, subset_fun; kwargs...)
end

function all_trajectories_id(search_tree, goal_state, subset_fun; verbose = false, max_trajectories = typemax(Int), beam_size = typemax(Int))
	trajectories = [[k] for (k,v) in search_tree if issubset(goal_state, v.state)]
	while any(!is_root(first(t), search_tree) for t in trajectories)
		trajectories = add_all_parents(search_tree, trajectories)
		before_beam = length(trajectories)
		trajectories = subset_fun(trajectories, beam_size)
		after_beam = length(trajectories)
		if verbose
			finished = sum(is_root(first(t), search_tree) for t in trajectories)
			println("Number of trajectories before beam: ",before_beam, " after beam: ", after_beam, " complete: ", finished)
		end
		trajectories = unique(trajectories)
	end
	trajectories = subset_fun(trajectories, max_trajectories)
	trajectories
end

function all_trajectories(search_tree, goal_state; kwargs...)
	trajectories = all_trajectories_id(search_tree, goal_state; kwargs...)
	map(trajectories) do trajectory
		[search_tree[id].state for id in trajectory]
	end
end


"""
	maxcover_subset(trajectories, n)

	Return subset of `n` trajectories which have maximally different states.
	The subset is found using greedy algorithm.
"""
function maxcover_subset(trajectories, n)
	trajectories = unique(trajectories)
	length(trajectories) ≤ n && return(trajectories)
	ts = [argmin(length, trajectories)]
	for i in 2:n 
		t = argmax(setdiff(trajectories, ts)) do t₁ # add trajectory which has maximum distance from the closest trrajectory
			minimum(length ∘ Base.Fix1(intersect, t₁), ts)
		end 
		push!(ts, t)
	end
	return(ts)

end


"""
	maxstates_subset(trajectories, n)

	Return subset of `n` trajectories which contain maximum number of states in the trajectory
"""
function maxstates_subset(trajectories, n)
	length(trajectories) ≤ n && return(trajectories)
	ts = [argmin(length, trajectories)]
	states = Set(first(ts))
	for i in 2:n 
		t = argmax(Base.Fix1(sum, ∉(states)), trajectories)
		union!(states, t)
		push!(ts, t)
	end
	return(ts)

end
