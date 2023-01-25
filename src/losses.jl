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

function L₂MiniBatch(pddld, problem::Problem, sol; goal_aware = true)
	sol.status != :success && return(UnsolvedL₂(sol, pddld, problem))
	L₂MiniBatch(pddld, problem, sol.trajectory; goal_aware)
end

function L₂MiniBatch(pddld, problem::Problem, trajectory::AbstractVector{<:State}; goal_aware = true)
	pddle = goal_aware ? add_goalstate(pddld, problem) : pddld
	L₂MiniBatch(pddle, trajectory)
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


function LₛMiniBatch(sol::PathSearchSolution, pddle, trajectory::AbstractVector{<:State})
	LₛMiniBatch(sol.search_tree, pddle, trajectory)
end

function LₛMiniBatch(search_tree::Dict{UInt64, <:PathNode}, pddle, trajectory::AbstractVector{<:State})
	# get indexes of the states on the solution path, which seems to be hashes of states 
	trajectory_id = hash.(trajectory)
	child₁ = descendants(search_tree, trajectory_id)
	# child₂ = descendants(sol, union(child₁, trajectory_id))
	ids = vcat(trajectory_id, child₁)
	path_cost = [search_tree[i].path_cost for i in ids]
	states = [search_tree[i].state for i in ids]
	# we want every state on the solution path to be smaller than  
	pm = [(i,j) for i in 1:length(trajectory_id) for j in length(trajectory_id)+1:length(ids)]
	H₊ = onehotbatch([i[2] for i in pm], 1:length(ids))
	H₋ = onehotbatch([i[1] for i in pm], 1:length(ids))

	LₛMiniBatch(batch(map(pddle, states)), H₊, H₋, path_cost, length(trajectory_id))
end

function LₛMiniBatch(pddld, domain::GenericDomain, problem::Problem, plan::AbstractVector{<:Julog.Term}; goal_aware = true)
	state = initstate(domain, problem)
	trajectory = SymbolicPlanners.simulate(StateRecorder(), domain, state, plan)
	LₛMiniBatch(pddld, domain, problem, trajectory; goal_aware)
end

function LₛMiniBatch(pddld, domain::GenericDomain, problem::Problem, trajectory::AbstractVector{<:State}; goal_aware = true)
	goal =  goalstate(domain, problem)
	pddle = goal_aware ? NeuroPlanner.add_goalstate(pddld, problem, goal) : pddld

	state = initstate(domain, problem)
	@assert issubset(trajectory[1].facts, state.facts)
	state = trajectory[1]
	spec = Specification(problem)
	spec = SymbolicPlanners.simplify_goal(spec, domain, state)

	stateids = Dict(state => (;id = 1, g = 0))
	I₊ = Vector{Int64}()
	I₋ = Vector{Int64}()

	for i in 1:(length(trajectory)-1)
		sᵢ, sⱼ = trajectory[i], trajectory[i+1]
		acts = available(domain, sᵢ)
		isempty(acts) && break 
		#add states to the the map
		next_states = map(acts) do act
			next_state = execute(domain, sᵢ, act; check=false)
			act_cost = get_cost(spec, domain, sᵢ, act, next_state)
			if next_state ∉ keys(stateids)
				stateids[next_state] = (;id = length(stateids) + 1,
					g = stateids[sᵢ].g + act_cost
					)
			end
			next_state
		end
		@assert sⱼ ∈ next_states
		next_states = setdiff(next_states, keys(stateids))

		for s in setdiff(keys(stateids), trajectory)
			push!(I₊, stateids[s].id)
			push!(I₋, stateids[sᵢ].id)			
		end
	end

	H₊ = onehotbatch(I₊, 1:length(stateids))
	H₋ = onehotbatch(I₋, 1:length(stateids))
	states = collect(keys(stateids))
	states = sort(states, lt = (i,j) -> stateids[i].id < stateids[j].id)
	@assert stateids[states[1]].id == 1
	@assert stateids[states[end]].id == length(stateids)
	path_cost = [stateids[s].g for s in states]
	LₛMiniBatch(batch(map(pddle, states)), H₊, H₋, path_cost, length(trajectory))
end


"""
LₛLoss(x, g, H₊, H₋)

Minimizes `L*` loss, We want ``f * H₋ .< f * H₊``, which means to minimize cases when ``f * H₋ .> f * H₊``
"""
struct LₛLoss end 

function lₛloss(model, x, g, H₊, H₋)
	g = reshape(g, 1, :)
	f = model(x) + g
	o = f * H₋ - f * H₊
	isempty(o) && return(zero(eltype(o)))
	mean(softplus.(o))
end
lₛloss(model, xy::LₛMiniBatch) = lₛloss(model, xy.x, xy.path_cost, xy.H₊, xy.H₋)
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


function LgbfsMiniBatch(sol, pddld, problem::Problem)
	sol.search_tree === nothing && error("solve the problem with `save_search=true` to keep the search tree")
	sol.status != :success && return(UnsolvedLgbfs(sol, pddld, problem))
	pddle = NeuroPlanner.initproblem(pddld, problem)[1]
	trajectory = sol.trajectory
	l = LₛMiniBatch(sol, pddle, trajectory)
	LgbfsMiniBatch(l.x, l.H₊, l.H₋, l.path_cost, l.sol_length)
end

function LgbfsMiniBatch(pddld, domain::GenericDomain, problem::Problem, trajectory; goal_aware = true)
	l = LₛMiniBatch(pddld, domain, problem, trajectory)
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

function lgbfsloss(model, x, g, H₊, H₋)
	f = model(x)
	o = f * H₋ .- f * H₊
	isempty(o) && return(zero(eltype(o)))
	mean(softplus.(o))
end

lgbfsloss(model, xy::LgbfsMiniBatch) = lgbfsloss(model, xy.x, xy.path_cost, xy.H₊, xy.H₋)
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
	H₊ = onehotbatch(1:n -1 , 1:n)
	H₋ = onehotbatch(2:n, 1:n)
	LRTMiniBatch(batch(map(pddle, trajectory)),
		H₊,
		H₋,
		collect(length(trajectory):-1:1),
		n
     )
end

function LRTMiniBatch(pddld, problem::Problem, sol; goal_aware = true)
	sol.status != :success && return(UnsolvedL₂(sol, pddld, problem))
	LRTMiniBatch(pddld, problem, sol.trajectory; goal_aware)
end

function LRTMiniBatch(pddld, problem::Problem, trajectory::AbstractVector{<:State}; goal_aware = true)
	pddle = goal_aware ? add_goalstate(pddld, problem) : pddld
	LRTMiniBatch(pddle, trajectory)
end


function LRTMiniBatch(pddld, domain::GenericDomain, problem::Problem, plan::AbstractVector{<:Julog.Term}; goal_aware = true)
	state = initstate(domain, problem)
	trajectory = SymbolicPlanners.simulate(StateRecorder(), domain, state, plan)
	LRTMiniBatch(pddld, problem, trajectory; goal_aware)
end

struct LRTLoss end 

function lrtloss(model, x, g, H₊, H₋)
	f = model(x)
	o = f * H₋ - f * H₊
	isempty(o) && return(zero(eltype(o)))
	mean(softplus.(o))
end

lrtloss(model, xy::LRTMiniBatch) = lrtloss(model, xy.x, xy.path_cost, xy.H₊, xy.H₋)
lrtloss(model, mb::NamedTuple{(:minibatch,:stats)}) = lrtloss(model, mb.minibatch)
(l::LRTLoss)(args...) = lrtloss(args...)


"""
descendants(sol, parents_id)

returns `descendants` of `parents_id` expanded during the search,
which are disjoint of `parents_id`, i.e.  
parents_id ∩ descendants(sol, parents_id) = ∅
"""
descendants(sol, parents_id::Vector) = descendants(sol, Set(parents_id))

function descendants(sol::PathSearchSolution, parents_id::Set)
	descendants(sol.search_tree, parents_id)
end

function descendants(search_tree::Dict{UInt64,<:PathNode}, parents_id::Set)
	childs = [s.id for s in values(search_tree) if s.parent_id ∈ parents_id]
	setdiff(childs, parents_id)
end

function getloss(name)
	name == "l2" && return((L₂Loss(), L₂MiniBatch))
	name == "l₂" && return((L₂Loss(), L₂MiniBatch))
	name == "lstar" && return((LₛLoss(), LₛMiniBatch))
	name == "lₛ" && return((LₛLoss(), LₛMiniBatch))
	name == "lgbfs" && return((LgbfsLoss(), LgbfsMiniBatch))
	name == "lrt" && return((LRTLoss(), LRTMiniBatch))
	error("unknown loss $(name)")
end