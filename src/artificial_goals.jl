using PDDL: get_facts

"""
	subgoals(problem)
	
	create all problems with goals containing a subset of predicates of the original goal.

"""
function subgoals(problem::GenericProblem)
	goal = problem.goal
	map(combinations(goal.args)) do args 
		@set problem.goal.args = args
	end
end

"""
	subset(args, i, n)

	create at most `n` subset goals of length `i`
"""
function subgoals(args::AbstractVector{<:Term}, i::Integer, n::Integer)
	gs = combinations(args, i)
	l = length(gs)
	if l > 100 * n 
		return([sample(args, i, replace = false) for _ in 1:n])
	elseif l > n
		goals = collect(gs)
		return(sample(goals, n, replace  = false))
	else 
		collect(gs)
	end
end

function subgoals(problem::GenericProblem, i::Integer, n::Integer)
	args = problem.goal.args
	map(subgoals(args, i, n)) do args 
		@set problem.goal.args = args
	end	
end

function subgoals(problem::GenericProblem, max_per_length::Integer)
	n = length(problem.goal.args)
	reduce(vcat, [subgoals(problem, i, max_per_length) for i in 1:n-1])
end

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
	new_goal = artificial_goal(trajectory[end], goal)
	new_facts = get_facts(new_goal)
	i = findfirst(i -> issubset(new_facts, get_facts(trajectory[i])), 1:length(trajectory))
	trajectory[1:i], plan[1:i-1], new_goal
end


"""
artificial_goal(target_goal::GenericState, goal::GenericState)

Create a substate of `target_goal` with similar predicates as `goal`,
such that the maximal facts between `goal` and `target_goal` are copied 
and the distribution of types is preserved.
"""
function artificial_goal(tgoal::GenericState, goal::GenericState)
	gf = get_facts(goal)
	gt = get_facts(tgoal)

	new_facts = intersect(gf, gt)
	gf = setdiff(gf, new_facts)	
	gt = collect(setdiff(gt, new_facts))

	gn = countmap([f.name for f in gf])
	if !isempty(gn)
		sampled_facts = mapreduce(vcat, keys(gn)) do k
			v = gn[k]
			tt = filter(x -> x.name == k, gt)
			sample(tt, min(v, length(tt)), replace = false)
		end
		new_facts = union(new_facts, sampled_facts)
	end
	GenericState(tgoal.types, Set(new_facts))
end

function artificial_goal(tgoal, goal)
	error("artificial_goal is impemented only for interpreted states, not compiled ones")
end