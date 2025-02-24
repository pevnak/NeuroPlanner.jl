
function LₛMiniBatchPossibleInequalities(pddld, domain::GenericDomain, problem::GenericProblem, plan::AbstractVector{<:Julog.Term}; goal_aware = true, max_branch = typemax(Int), plot_dict=nothing)
	state = initstate(domain, problem)
	trajectory = SymbolicPlanners.simulate(StateRecorder(), domain, state, plan)
	LₛMiniBatchPossibleInequalities(pddld, domain, problem, trajectory; goal_aware, max_branch, plot_dict)
end

function LₛMiniBatchPossibleInequalities(pddld, domain::GenericDomain, problem::GenericProblem, trajectory::AbstractVector{<:GenericState}; goal_aware = true, max_branch = typemax(Int), plot_dict=nothing)
	LₛMiniBatchPossibleInequalities(pddld, domain, problem, nothing, trajectory; goal_aware, max_branch, plot_dict)
end

function LₛMiniBatchPossibleInequalities(pddld, domain::GenericDomain, problem::GenericProblem, st::Union{Nothing,RSearchTree}, trajectory::AbstractVector{<:GenericState}; goal_aware = true, max_branch = typemax(Int), plot_dict=nothing)
	pddle = goal_aware ? NeuroPlanner.add_goalstate(pddld, problem) : pddld
	state = trajectory[1]
	spec = Specification(problem)
	lmcut = LM_CutHeuristic()

	
	lmvals = Dict(hash(state) => lmcut(domain, state, spec))

	I₊ = Vector{Int64}()
	I₋ = Vector{Int64}()
	
	max_time = 30
	astar = AStarPlanner(lmcut; save_search=true, save_search_order=true, max_time)
	sol = astar(domain, state, PDDL.get_goal(problem))
	@show sol.status
	sol.status == :failure && error("astar could not find solution")
	if sol.status != :success
		return nothing
	end
	
	
	frontier_to_exapnd = collect(keys(sol.search_frontier))
	for hfrontier in frontier_to_exapnd
		parent_state = sol.search_tree[hfrontier]
		SymbolicPlanners.expand!(astar, lmcut, parent_state, sol.search_tree, sol.search_frontier, domain, spec)
	end
	@show length(values(sol.search_tree))
	states = values(sol.search_tree)

	trajectory = sol.trajectory
	off_traj = Vector{SymbolicPlanners.PathNode}()
	id_counter = 1
	for (i,path_state) in enumerate(states)
		if path_state.state in trajectory

			continue
		end
		push!(off_traj, path_state)
		lmvals[path_state.id] = lmcut(domain, path_state.state, spec)
	end
	off_traj_offset = length(trajectory)
	for (i, traj_state) in enumerate(trajectory)
		for (j, off_traj_state) in enumerate(off_traj)
			if off_traj_state.path_cost + lmvals[off_traj_state.id] <= i-1
				#println("skipped")
				continue
			end
			push!(I₊, j+off_traj_offset)
			push!(I₋, i)
		end
	end
	n_states = length(states)
	
	if !isnothing(plot_dict)
		plot_dict[:sol] = sol
	end
	#if states[stateids[s]].g + lmvals[s] <= states[stateids[hsⱼ]].g
	if isempty(I₊)
		H₊ = onehotbatch([], 1:n_states)
		H₋ = onehotbatch([], 1:n_states)
	else
		H₊ = onehotbatch(I₊, 1:n_states)
		H₋ = onehotbatch(I₋, 1:n_states)
	end
	path_cost = [s.path_cost for s in states]
	inner_type = typeof(pddle(first(states).state))
	batch_vec = Vector{inner_type}()
	for (i,state) in enumerate(trajectory)
		push!(batch_vec, pddle(state))
		if !isnothing(plot_dict)
			ids = get!(Dict{UInt64, Int64}, plot_dict, :ids)
			push!(ids,	hash(state) => id_counter)				
			id_counter+=1

		end
	end
	for(i,path_state) in enumerate(off_traj)
		push!(batch_vec, pddle(path_state.state))
		if !isnothing(plot_dict)
			ids = get!(Dict{UInt64, Int64}, plot_dict, :ids)
			push!(ids, path_state.id => i+off_traj_offset)
		end
	end
	#x = batch(inner_type[pddle(s.state) for s in states])
	x = batch(batch_vec)
	LₛMiniBatch(x, H₊, H₋, path_cost, length(trajectory))
end

function cost_state_to_state(first_state::GenericState, second_state::GenericState, domain, spec) 
	acts = available(domain, first_state)
	for act in acts
		if execute(domain, first_state, act; check=false) == second_state
			return get_cost(spec, domain, first_state, act, second_state)
		end
	end
	prinln("No connecting action found")
	return -1
end


#############
#	Binary Classification Losses
#############

struct BinClassBatch{X,H,Y} <: AbstractMinibatch
	x::X
	H::H
	path_cost::Y
	sol_length::Int64
end

function BinClassBatch(pddld, domain::GenericDomain, problem::GenericProblem, plan::AbstractVector{<:Julog.Term}; goal_aware = true, max_branch = typemax(Int), plot_dict=nothing)
	state = initstate(domain, problem)
	trajectory = SymbolicPlanners.simulate(StateRecorder(), domain, state, plan)
	BinClassBatch(pddld, domain, problem, trajectory; goal_aware, max_branch, plot_dict)
end

function BinClassBatch(pddld, domain::GenericDomain, problem::GenericProblem, trajectory::AbstractVector{<:GenericState}; goal_aware = true, max_branch = typemax(Int), plot_dict=nothing)
	BinClassBatch(pddld, domain, problem, nothing, trajectory; goal_aware, max_branch, plot_dict)
end

function BinClassBatch(pddld, domain::GenericDomain, problem::GenericProblem, st::Union{Nothing,RSearchTree}, trajectory::AbstractVector{<:GenericState}; goal_aware = true, max_branch = typemax(Int), plot_dict=nothing)
	pddle = goal_aware ? NeuroPlanner.add_goalstate(pddld, problem) : pddld
	state = trajectory[1]
	spec = Specification(problem)
	lmcut = LM_CutHeuristic()
	trajectory_length = length(trajectory)

	stateids = Dict(hash(state) => 1)
	states = [(g = 0, state = state)]
	lmvals = Dict(hash(state) => lmcut(domain, state, spec))
	I = Vector{Int64}()
	push!(I, 2)

	htrajectory = hash.(trajectory)
	for i in 1:(length(trajectory)-1)
		if !isnothing(plot_dict)
			depth_indexs = get!(Vector{Int}, plot_dict, :depth)
			push!(depth_indexs, length(stateids))
		end
		sᵢ, sⱼ = trajectory[i], trajectory[i+1]
		hsⱼ = hash(sⱼ)
		gᵢ = states[stateids[hash(sᵢ)]].g
		next_states = _next_states(domain, problem, sᵢ, st)
		if length(next_states) > max_branch # limit the branching factor is not excessive
			ii = findall(s -> s.state == sⱼ, next_states)
			ii = union(ii, sample(1:length(next_states), max_branch, replace = false))
			next_states = next_states[ii]
		end
		isempty(next_states) && error("the inner node is not in the search tree")
		for next_state in next_states
			act = next_state.parent_action
			act_cost = get_cost(spec, domain, sᵢ, act, next_state.state)
			next_state.id ∈ keys(stateids) && continue
			stateids[next_state.id] = length(stateids) + 1
			push!(states, (;g = gᵢ + act_cost, state = next_state.state))
			classification = next_state.state in trajectory ? 2 : 1
			push!(I, classification)
			if next_state.state ∈ trajectory && !isnothing(plot_dict)
				traj_indexes = get!(Vector{Int}, plot_dict, :traj)
				push!(traj_indexes, stateids[next_state.id])
			end
		end
	end
	@show I
	if isempty(I)
		H = onehotbatch([], 1:2)
	else
		H = onehotbatch(I, 1:2)
	end
	
	path_cost = [s.g for s in states]
	inner_type = typeof(pddle(first(states).state))
	x = batch(inner_type[pddle(s.state) for s in states])
	BinClassBatch(x,H, path_cost, length(trajectory))
end


function binclassloss(model, x, H)
	Flux.Losses.logitcrossentropy(model(x), H)
end
binclassloss(model, xy::BinClassBatch) = binclassloss(model, xy.x, xy.H)
binclassloss(model, mb::NamedTuple{(:minibatch,:stats)}) = binclassloss(model, mb.minibatch)

