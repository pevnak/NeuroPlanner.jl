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

function L₂MiniBatch(pddle, trajectory::AbstractVector{<:State})
	L₂MiniBatch(batch(map(pddle, trajectory)),
     collect(length(trajectory):-1:1),
     )
end


function L₂MiniBatch(pddld, domain::GenericDomain, problem::Problem, trajectory::AbstractVector{<:State}; goal_aware = true)
	pddle = goal_aware ? add_goalstate(pddld, problem) : pddld
	L₂MiniBatch(pddle, trajectory)
end

function L₂MiniBatch(pddld, domain::GenericDomain, problem::Problem, st::Union{Nothing,RSearchTree}, trajectory::AbstractVector{<:State}; goal_aware = true)
	L₂MiniBatch(pddld, domain, problem, trajectory; goal_aware)
end

function L₂MiniBatch(pddld, domain::GenericDomain, problem::Problem, plan::AbstractVector{<:Julog.Term}; goal_aware = true)
	state = initstate(domain, problem)
	trajectory = SymbolicPlanners.simulate(StateRecorder(), domain, state, plan)
	L₂MiniBatch(pddld, problem, trajectory; goal_aware)
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


function LₛMiniBatch(sol, pddld, problem::Problem)
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

function LₛMiniBatch(pddld, domain::GenericDomain, problem::Problem, plan::AbstractVector{<:Julog.Term}; goal_aware = true)
	state = initstate(domain, problem)
	trajectory = SymbolicPlanners.simulate(StateRecorder(), domain, state, plan)
	LₛMiniBatch(pddld, domain, problem, trajectory; goal_aware)
end

function LₛMiniBatch(pddld, domain::GenericDomain, problem::Problem, trajectory::AbstractVector{<:State}; goal_aware = true)
	LₛMiniBatch(pddld, domain, problem, nothing, trajectory; goal_aware)
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

function LₛMiniBatch(pddld, domain::GenericDomain, problem::Problem, st::Union{Nothing,RSearchTree}, trajectory::AbstractVector{<:State}; goal_aware = true)
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


"""
LₛLoss(x, g, H₊, H₋)

Minimizes `L*` loss, We want ``f * H₋ .< f * H₊``, which means to minimize cases when ``f * H₋ .> f * H₊``
"""
struct LₛLoss end 

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
(l::LₛLoss)(args...) = lₛloss(args...)

#############
#	Lstar Losses
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

function LgbfsMiniBatch(pddld, domain::GenericDomain, problem::Problem, trajectory; goal_aware = true)
	l = LₛMiniBatch(pddld, domain, problem, trajectory; goal_aware)
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


struct LRTMiniBatch{X,H,Y}
	x::X 
	H₊::H 
	H₋::H 
	path_cost::Y
	sol_length::Int64
end

function LRTMiniBatch(pddle, trajectory::AbstractVector{<:State})
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

function LRTMiniBatch(pddld, domain::GenericDomain, problem::Problem, plan::AbstractVector{<:Julog.Term}; goal_aware = true)
	state = initstate(domain, problem)
	trajectory = SymbolicPlanners.simulate(StateRecorder(), domain, state, plan)
	LRTMiniBatch(pddld, domain, problem, trajectory; goal_aware)
end


function LRTMiniBatch(pddld, domain::GenericDomain, problem::Problem, st::Union{Nothing,RSearchTree}, trajectory::AbstractVector{<:State}; goal_aware = true)
	LRTMiniBatch(pddld, domain, problem, trajectory; goal_aware)
end

function LRTMiniBatch(pddld, domain::GenericDomain, problem::Problem, trajectory; goal_aware = true)
	pddle = goal_aware ? NeuroPlanner.add_goalstate(pddld, problem) : pddld
	LRTMiniBatch(pddle, trajectory)
end

struct LRTLoss end 

function lrtloss(model, x, g, H₊, H₋, surrogate = softplus)
	f = model(x)
	o = f * H₋ - f * H₊
	isempty(o) && return(zero(eltype(o)))
	mean(surrogate.(o))
end

lrtloss(model, xy::LRTMiniBatch) = lrtloss(model, xy.x, xy.path_cost, xy.H₊, xy.H₋)
lrtloss(model, xy::LRTMiniBatch, surrogate) = lrtloss(model, xy.x, xy.path_cost, xy.H₊, xy.H₋, surrogate)
lrtloss(model, mb::NamedTuple{(:minibatch,:stats)}) = lrtloss(model, mb.minibatch)
(l::LRTLoss)(args...) = lrtloss(args...)


########
#	dispatch for loss function
########
loss(model, xy::L₂MiniBatch,surrogate=softplus) = l₂loss(model, xy, surrogate)
loss(model, xy::LₛMiniBatch,surrogate=softplus) = lₛloss(model, xy, surrogate)
loss(model, xy::LgbfsMiniBatch,surrogate=softplus) = lgbfsloss(model, xy, surrogate)
loss(model, xy::LRTMiniBatch,surrogate=softplus) = lrtloss(model, xy, surrogate)
loss(model, xy::Tuple,surrogate=softplus) = sum(map(x -> lossfun(model, x), xy), surrogate)


function minibatchconstructor(name)
	name == "l2" && return(L₂MiniBatch)
	name == "l₂" && return(L₂MiniBatch)
	name == "lstar" && return(LₛMiniBatch)
	name == "lₛ" && return(LₛMiniBatch)
	name == "lgbfs" && return(LgbfsMiniBatch)
	name == "lrt" && return(LRTMiniBatch)
	error("unknown loss $(name)")
end