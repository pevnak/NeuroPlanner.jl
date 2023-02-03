using PDDL: get_facts
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

"""
trajectory, plan, new_goal = artificial_goal(domain, problem, trajectory, goal)

finds a new goal in trajectory with predicates similar to goals 
"""
function artificial_goal(domain, problem, trajectory, plan, goal = goalstate(domain, problem))
	tgoal = last(trajectory)
	gf = get_facts(goal)
	gn = countmap([f.name for f in gf])
	gt = collect(get_facts(tgoal))
	new_facts = mapreduce(vcat, keys(gn)) do k
		v = gn[k]
		tt = filter(x -> x.name == k, gt)
		sample(tt, min(v, length(tt)), replace = false)
	end
	new_goal = GenericState(tgoal.types, Set(new_facts))

	i = findfirst(i -> issubset(new_facts, get_facts(trajectory[i])), 1:length(trajectory))
	trajectory[1:i], plan[1:i-1], new_goal
end