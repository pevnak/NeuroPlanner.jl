using Parameters
using SymbolicPlanners: PathNode
using DataStructures: Queue
using OneHotArrays: onehotbatch

#############
#	L2 Loss
#############

struct L₂MiniBatch{X,Y}
	x::X 
	y::Y
end

struct UnsolvedL₂{S,D,P}
	sol::S 
	pddld::D
	problem::P 
end

function L₂MiniBatch(pddle, trajectory::AbstractVector{<:GenericState})
	L₂MiniBatch(batch(map(pddle, trajectory)),
     collect(length(trajectory):-1:1),
     )
end


function L₂MiniBatch(pddld, domain::GenericDomain, problem::GenericProblem, trajectory::AbstractVector{<:GenericState}; goal_aware = true, max_branch = typemax(Int))
	pddle = goal_aware ? add_goalstate(pddld, problem) : pddld
	L₂MiniBatch(pddle, trajectory)
end

function L₂MiniBatch(pddld, domain::GenericDomain, problem::GenericProblem, st::Union{Nothing,RSearchTree}, trajectory::AbstractVector{<:GenericState}; goal_aware = true, max_branch = typemax(Int))
	L₂MiniBatch(pddld, domain, problem, trajectory; goal_aware, max_branch)
end

function L₂MiniBatch(pddld, domain::GenericDomain, problem::GenericProblem, plan::AbstractVector{<:Julog.Term}; goal_aware = true, max_branch = typemax(Int))
	state = initstate(domain, problem)
	trajectory = SymbolicPlanners.simulate(StateRecorder(), domain, state, plan)
	L₂MiniBatch(pddld, domain, problem, trajectory; goal_aware, max_branch)
end


function prepare_minibatch(mb::UnsolvedL₂)
	@unpack sol, problem, pddld = mb
	trajectory = artificial_trajectory(sol)
	goal = trajectory[end]
	pddle = NeuroPlanner.add_goalstate(pddld, problem, goal)
	L₂MiniBatch(sol, pddle, trajectory)
end 

struct L₂Loss end 
l₂loss(model, x, y) = Flux.Losses.mse(vec(model(x)), y)
l₂loss(model, xy::L₂MiniBatch) = l₂loss(model, xy.x, xy.y)
l₂loss(model, xy::L₂MiniBatch, surrogate) = l₂loss(model, xy.x, xy.y)
l₂loss(model, mb::NamedTuple{(:minibatch,:stats)}) = l₂loss(model, mb.minibatch)
(l::L₂Loss)(args...) = l₂loss(args...)

#############
#	Lstar Losses as described in  Chrestien, Leah, et al. "Optimize planning heuristics to rank, not to estimate cost-to-goal." Advances in Neural Information Processing Systems 36 (2024).
#############
struct LₛMiniBatch{X,H,Y}
	x::X 
	H₊::H 
	H₋::H 
	path_cost::Y
	sol_length::Int64
end

struct UnsolvedLₛ{S,D,P}
	sol::S 
	pddld::D
	problem::P 
end


function LₛMiniBatch(sol, pddld, problem::GenericProblem)
	sol.search_tree === nothing && error("solve the problem with `save_search=true` to keep the search tree")
	sol.status != :success && return(UnsolvedLₛ(sol, pddld, problem))
	pddle = NeuroPlanner.initproblem(pddld, problem)[1]
	trajectory = sol.trajectory
	LₛMiniBatch(sol, pddle, trajectory)
end

function prepare_minibatch(mb::UnsolvedLₛ)
	@unpack sol, problem, pddld = mb
	trajectory = artificial_trajectory(sol)
	goal = trajectory[end]
	pddle = NeuroPlanner.add_goalstate(pddld, problem, goal)
	LₛMiniBatch(sol, pddle, trajectory)
end

function LₛMiniBatch(pddld, domain::GenericDomain, problem::GenericProblem, plan::AbstractVector{<:Julog.Term}; goal_aware = true, max_branch = typemax(Int), plot_dict=nothing)
	state = initstate(domain, problem)
	trajectory = SymbolicPlanners.simulate(StateRecorder(), domain, state, plan)
	LₛMiniBatch(pddld, domain, problem, trajectory; goal_aware, max_branch, plot_dict)
end

function LₛMiniBatch(pddld, domain::GenericDomain, problem::GenericProblem, trajectory::AbstractVector{<:GenericState}; goal_aware = true, max_branch = typemax(Int), plot_dict=nothing)
	LₛMiniBatch(pddld, domain, problem, nothing, trajectory; goal_aware, max_branch, plot_dict)
end

_next_states(domain, problem, sᵢ, st::RSearchTree) = st.st[hash(sᵢ)]
function _next_states(domain, problem, sᵢ, st::Nothing) 
	acts = available(domain, sᵢ)
	isempty(acts) && return([])
	hsᵢ = hash(sᵢ)
	map(acts) do act
		state = PDDL.execute(domain, sᵢ, act; check=false)
		(;state,
		  id = hash(state),
		  parent_action = act,
		  parent_id = hsᵢ
		 )
	end
end

function LₛMiniBatch(pddld, domain::GenericDomain, problem::GenericProblem, st::Union{Nothing,RSearchTree}, trajectory::AbstractVector{<:GenericState}; goal_aware = true, max_branch = typemax(Int), plot_dict=nothing)
	pddle = goal_aware ? NeuroPlanner.add_goalstate(pddld, problem) : pddld
	state = trajectory[1]
	spec = Specification(problem)

	stateids = Dict(hash(state) => 1)
	states = [(g = 0, state = state)]
	I₊ = Vector{Int64}()
	I₋ = Vector{Int64}()

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
			if next_state.state ∈ trajectory && !isnothing(plot_dict)
				traj_indexes = get!(Vector{Int}, plot_dict, :traj)
				push!(traj_indexes, stateids[next_state.id])
			end
		end
		@assert hash(sⱼ) ∈ keys(stateids)
		open_set = setdiff(keys(stateids), htrajectory)

		for s in open_set
			#TODO sem dat nerovnost
			push!(I₊, stateids[s])
			push!(I₋, stateids[hsⱼ])
		end
	end
	if isempty(I₊)
		H₊ = onehotbatch([], 1:length(stateids))
		H₋ = onehotbatch([], 1:length(stateids))
	else
		H₊ = onehotbatch(I₊, 1:length(stateids))
		H₋ = onehotbatch(I₋, 1:length(stateids))
	end
	path_cost = [s.g for s in states]
	inner_type = typeof(pddle(first(states).state))
	x = batch(inner_type[pddle(s.state) for s in states])
	LₛMiniBatch(x, H₊, H₋, path_cost, length(trajectory))
end

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

function lₛloss(model, x, g, H₊, H₋, surrogate=softplus)
	g = reshape(g, 1, :)
	f = model(x) + g
	o = f * H₋ - f * H₊
	isempty(o) && return(zero(eltype(o)))
	mean(surrogate.(o))
end
lₛloss(model, xy::LₛMiniBatch) = lₛloss(model, xy.x, xy.path_cost, xy.H₊, xy.H₋)
lₛloss(model, xy::LₛMiniBatch, surrogate) = lₛloss(model, xy.x, xy.path_cost, xy.H₊, xy.H₋, surrogate)
lₛloss(model, mb::NamedTuple{(:minibatch,:stats)}) = lₛloss(model, mb.minibatch)

#############
#	Binary Classification Losses
#############

struct BinClassBatch{X,H,Y}
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

#############
#	LGBFS Losses
#############
struct LgbfsMiniBatch{X,H,Y}
	x::X 
	H₊::H 
	H₋::H 
	path_cost::Y
	sol_length::Int64
end

struct UnsolvedLgbfs{S,D,P}
	sol::S 
	pddld::D
	problem::P 
end

function LgbfsMiniBatch(pddld, domain::GenericDomain, problem::GenericProblem, trajectory; goal_aware = true, max_branch = typemax(Int))
	l = LₛMiniBatch(pddld, domain, problem, trajectory; goal_aware, max_branch)
	LgbfsMiniBatch(l.x, l.H₊, l.H₋, l.path_cost, l.sol_length)	
end

function prepare_minibatch(mb::UnsolvedLgbfs)
	@unpack sol, problem, pddld = mb
	trajectory = artificial_trajectory(sol)
	goal = trajectory[end]
	pddle = NeuroPlanner.add_goalstate(pddld, problem, goal)
	l = LₛMiniBatch(pddle, sol, trajectory)
	LgbfsMiniBatch(l.x, l.H₊, l.H₋, l.path_cost, l.sol_length)
end



#############
#	LgbfsLoss Losses as described in  Chrestien, Leah, et al. "Optimize planning heuristics to rank, not to estimate cost-to-goal." Advances in Neural Information Processing Systems 36 (2024).
#############

"""
LgbfsLoss(x, g, H₊, H₋)

Minimizes `L*`-like loss for the gbfs search. We want ``f * H₋ .< f * H₊``, which means to minimize cases when ``f * H₋ .> f * H₊``
"""
struct LgbfsLoss end 

(l::LgbfsLoss)(args...) = lgbfsloss(args...)

function lgbfsloss(model, x, g, H₊, H₋, surrogate=softplus)
	f = model(x)
	o = f * H₋ .- f * H₊
	isempty(o) && return(zero(eltype(o)))
	mean(surrogate.(o))
end

lgbfsloss(model, xy::LgbfsMiniBatch) = lgbfsloss(model, xy.x, xy.path_cost, xy.H₊, xy.H₋)
lgbfsloss(model, xy::LgbfsMiniBatch, surrogate) = lgbfsloss(model, xy.x, xy.path_cost, xy.H₊, xy.H₋, surrogate)
lgbfsloss(model, mb::NamedTuple{(:minibatch,:stats)}) = lgbfsloss(model, mb.minibatch)


#########
#	LRT loss enforcing ordering on the trajectory
#   Caelan Reed Garrett, Leslie Pack Kaelbling, and Tomás Lozano-Pérez. Learning to rank for synthesizing planning heuristics. page 3089–3095, 2016.
#########
struct LRTMiniBatch{X,H,Y}
	x::X 
	H₊::H 
	H₋::H 
	path_cost::Y
	sol_length::Int64
end

function LRTMiniBatch(pddle, trajectory::AbstractVector{<:GenericState})
	n = length(trajectory)
	if n < 2
		H₊ = onehotbatch([], 1:n)
		H₋ = onehotbatch([], 1:n)
	else
		H₊ = onehotbatch(1:n -1 , 1:n)
		H₋ = onehotbatch(2:n, 1:n)
	end
	LRTMiniBatch(batch(map(pddle, trajectory)),
		H₊,
		H₋,
		collect(length(trajectory):-1:1),
		n
     )
end

function LRTMiniBatch(pddld, domain::GenericDomain, problem::GenericProblem, plan::AbstractVector{<:Julog.Term}; goal_aware = true, max_branch = typemax(Int))
	state = initstate(domain, problem)
	trajectory = SymbolicPlanners.simulate(StateRecorder(), domain, state, plan)
	LRTMiniBatch(pddld, domain, problem, trajectory; goal_aware, max_branch)
end


function LRTMiniBatch(pddld, domain::GenericDomain, problem::GenericProblem, st::Union{Nothing,RSearchTree}, trajectory::AbstractVector{<:GenericState}; goal_aware = true, max_branch = typemax(Int))
	LRTMiniBatch(pddld, domain, problem, trajectory; goal_aware, max_branch)
end

function LRTMiniBatch(pddld, domain::GenericDomain, problem::GenericProblem, trajectory; goal_aware = true, max_branch = typemax(Int))
	pddle = goal_aware ? NeuroPlanner.add_goalstate(pddld, problem) : pddld
	LRTMiniBatch(pddle, trajectory)
end

function lrtloss(model, x, g, H₊, H₋, surrogate = softplus)
	f = model(x)
	o = f * H₋ - f * H₊
	isempty(o) && return(zero(eltype(o)))
	mean(surrogate.(o))
end

lrtloss(model, xy::LRTMiniBatch, surrogate = softplus) = lrtloss(model, xy.x, xy.path_cost, xy.H₊, xy.H₋)


########
#	BellmanLoss
########
"""
A bellman loss function combining inequalities in the expanded actions with L2 loss 
as proposed in 
Ståhlberg, Simon, Blai Bonet, and Hector Geffner. "Learning Generalized Policies without Supervision Using GNNs.", 2022
Equation (15)
"""
struct BellmanMiniBatch{X,Y}
	x::X 
	path_cost::Y
	trajectory_states::Vector{Int}
	child_states::Vector{Vector{Int}}
	cost_to_goal::Vector{Float32}
	sol_length::Int64
end

function BellmanMiniBatch(pddld, domain::GenericDomain, problem::GenericProblem, st::Union{Nothing,RSearchTree}, trajectory::AbstractVector{<:GenericState}; goal_aware = true, max_branch = typemax(Int))
	pddle = goal_aware ? NeuroPlanner.add_goalstate(pddld, problem) : pddld
	state = trajectory[1]
	spec = Specification(problem)

	stateids = Dict(hash(state) => 1)
	states = [(g = 0, state = state)]
	trajectory_states = Vector{Int}()
	child_states = Vector{Vector{Int}}()

	htrajectory = hash.(trajectory)
	for i in 1:(length(trajectory)-1)
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
		end
		@assert hash(sⱼ) ∈ keys(stateids) "next state on the trajectory is not in the oppen list"

		push!(child_states, [stateids[x.id] for x in next_states])
	end
	path_cost = [s.g for s in states]
	x = batch([pddle(s.state) for s in states])
	trajectory_states = [stateids[hash(s)] for s in trajectory]
	cost_to_goal = Float32.(collect(length(trajectory):-1:1))
	BellmanMiniBatch(x, path_cost, trajectory_states, child_states, cost_to_goal, length(trajectory))
end

function BellmanMiniBatch(pddld, domain::GenericDomain, problem::GenericProblem, plan::AbstractVector{<:Julog.Term}; goal_aware = true, max_branch = typemax(Int))
	state = initstate(domain, problem)
	trajectory = SymbolicPlanners.simulate(StateRecorder(), domain, state, plan)
	BellmanMiniBatch(pddld, domain, problem, trajectory; goal_aware, max_branch)
end

function BellmanMiniBatch(pddld, domain::GenericDomain, problem::GenericProblem, trajectory::AbstractVector{<:GenericState}; goal_aware = true, max_branch = typemax(Int))
	BellmanMiniBatch(pddld, domain, problem, nothing, trajectory; goal_aware, max_branch)
end


function bellmanloss(model, x, g, trajectory_states, child_states, cost_to_goal, surrogate=softplus)
	f = model(x)
	v = f[trajectory_states]
	vs = cost_to_goal
	reg = mean(max.(0, vs - v) + max.(0, v - 2*vs))
	isempty(child_states) && return(reg)
	h = map(1:length(child_states)) do i 
		max(1 + minimum(f[child_states[i]]) - f[trajectory_states[i]], 0)
	end
	mean(h) + reg 
end

bellmanloss(model, xy::BellmanMiniBatch, surrogate = softplus) = bellmanloss(model, xy.x, xy.path_cost, xy.trajectory_states, xy.child_states, xy.cost_to_goal)

########
#	dispatch for loss function
########
loss(model, xy::L₂MiniBatch,surrogate=softplus) = l₂loss(model, xy, surrogate)
loss(model, xy::LₛMiniBatch,surrogate=softplus) = lₛloss(model, xy, surrogate)
loss(model, xy::LgbfsMiniBatch,surrogate=softplus) = lgbfsloss(model, xy, surrogate)
loss(model, xy::LRTMiniBatch,surrogate=softplus) = lrtloss(model, xy, surrogate)
loss(model, xy::BellmanMiniBatch,surrogate=softplus) = bellmanloss(model, xy, surrogate)
loss(model, xy::Tuple,surrogate=softplus) = sum(map(x -> lossfun(model, x), xy), surrogate)
loss(model, xy::BinClassBatch, surrogate=softplus) = binclassloss(model, xy)


function minibatchconstructor(name)
	name == "l2" && return(L₂MiniBatch)
	name == "l₂" && return(L₂MiniBatch)
	name == "lstar" && return(LₛMiniBatch)
	name == "lₛ" && return(LₛMiniBatch)
	name == "lgbfs" && return(LgbfsMiniBatch)
	name == "lrt" && return(LRTMiniBatch)
	name == "bellman" && return(BellmanMiniBatch)
	name == "levinloss" && return(LevinMiniBatch)
	name == "newlstar" && return(LₛMiniBatchPossibleInequalities)
	name == "binclass" && return(BinClassBatch)
	error("unknown loss $(name)")
end