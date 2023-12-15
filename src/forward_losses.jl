using Parameters
using SymbolicPlanners: PathNode
using OneHotArrays: onehotbatch
#############
#	L2 Losses
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
#	Lstar Losses
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

function LₛMiniBatch(pddld, domain::GenericDomain, problem::GenericProblem, plan::AbstractVector{<:Julog.Term}; goal_aware = true, max_branch = typemax(Int))
	state = initstate(domain, problem)
	trajectory = SymbolicPlanners.simulate(StateRecorder(), domain, state, plan)
	LₛMiniBatch(pddld, domain, problem, trajectory; goal_aware, max_branch)
end

function LₛMiniBatch(pddld, domain::GenericDomain, problem::GenericProblem, trajectory::AbstractVector{<:GenericState}; goal_aware = true, max_branch = typemax(Int))
	LₛMiniBatch(pddld, domain, problem, nothing, trajectory; goal_aware, max_branch)
end

_next_states(domain, problem, sᵢ, st::RSearchTree) = st.st[hash(sᵢ)]
function _next_states(domain, problem, sᵢ, st::Nothing) 
	acts = available(domain, sᵢ)
	isempty(acts) && return([])
	hsᵢ = hash(sᵢ)
	map(acts) do act
		state = execute(domain, sᵢ, act; check=false)
		(;state,
		  id = hash(state),
		  parent_action = act,
		  parent_id = hsᵢ
		 )
	end
end

function LₛMiniBatch(pddld, domain::GenericDomain, problem::GenericProblem, st::Union{Nothing,RSearchTree}, trajectory::AbstractVector{<:GenericState}; goal_aware = true, max_branch = typemax(Int))
	pddle = goal_aware ? NeuroPlanner.add_goalstate(pddld, problem) : pddld
	state = trajectory[1]
	spec = Specification(problem)

	stateids = Dict(hash(state) => 1)
	states = [(g = 0, state = state)]
	I₊ = Vector{Int64}()
	I₋ = Vector{Int64}()

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
		@assert hash(sⱼ) ∈ keys(stateids)
		open_set = setdiff(keys(stateids), htrajectory)

		for s in open_set
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
	x = batch([pddle(s.state) for s in states])
	LₛMiniBatch(x, H₊, H₋, path_cost, length(trajectory))
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


function minibatchconstructor(name)
	name ∈ ("l2","l₂")  && return(L₂MiniBatch)
	name == "backl2"  && return(BackwardL2)
	name ∈  ("lstar", "lₛ") && return(LₛMiniBatch)
	name == "backlstar" && return(BackwardLₛMiniBatch)
	name == "lgbfs" && return(LgbfsMiniBatch)
	name == "backlgbfs" && return(BackwardLgbfsMiniBatch)
	name == "lrt" && return(LRTMiniBatch)
	name == "backlrt" && return(BackwardLRTMiniBatch)

	# some prior art that had to be checked
	name == "bellman" && return(BellmanMiniBatch)
	name == "levinloss" && return(LevinMiniBatch)
	error("unknown loss $(name)")
end