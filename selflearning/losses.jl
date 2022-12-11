using Parameters
include("artificial_goal.jl")
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

function L₂MiniBatch(sol, pddld, problem::Problem)
	sol.status != :success && return(UnsolvedL₂(sol, pddld, problem))
	pddle = NeuroPlanner.initproblem(pddld, problem)[1]
	L₂MiniBatch(pddle, sol, trajectory)
end

function L₂MiniBatch(sol, pddle, trajectory::AbstractVector{<:State})
   L₂MiniBatch(batch(map(pddle, trajectory)),
     collect(length(trajectory):-1:1),
     )
end

function prepare_minibatch(mb::UnsolvedL₂)
	@unpack sol, problem, pddld = mb
	trajectory = artificial_trajectory(sol)
	goal = trajectory[end]
	pddle = NeuroPlanner.add_goalstate(pddld, problem, goal)
	L₂MiniBatch(sol, pddle, trajectory)
end 

struct L₂loss end 
l₂loss(model, x, y) = Flux.Losses.mse(vec(model(x)), y)
l₂loss(model, xy::L₂MiniBatch) = l₂loss(model, xy.x, xy.y)
l₂loss(model, mb::NamedTuple{(:minibatch,:stats)}) = l₂loss(model, mb.minibatch)


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


function LₛMiniBatch(sol, pddle, trajectory::AbstractVector{<:State})
	# get indexes of the states on the solution path, which seems to be hashes of states 
	trajectory_id = hash.(trajectory)
	child₁ = descendants(sol, trajectory_id)
	# child₂ = descendants(sol, union(child₁, trajectory_id))
	ids = vcat(trajectory_id, child₁)
	path_cost = [sol.search_tree[i].path_cost for i in ids]
	states = [sol.search_tree[i].state for i in ids]
	# we want every state on the solution path to be smaller than  
	pm = [(i,j) for i in 1:length(trajectory_id) for j in length(trajectory_id)+1:length(ids)]
	H₊ = onehotbatch([i[2] for i in pm], 1:length(ids))
	H₋ = onehotbatch([i[1] for i in pm], 1:length(ids))

	LₛMiniBatch(batch(map(pddle, states)), H₊, H₋, path_cost, length(trajectory_id))
end

"""
LₛLoss(x, g, H₊, H₋)

Minimizes `L*` loss, We want ``f * H₋ .< f * H₊``, which means to minimize cases when ``f * H₋ .> f * H₊``
"""
struct LₛLoss end 

function lₛloss(model, x, g, H₊, H₋)
	g = reshape(g, 1, :)
	f = model(x) .+ g
	o = f * H₋ .- f * H₊
	isempty(o) && return(zero(eltype(o)))
	mean(softplus.(o))
end
lₛloss(model, xy::LₛMiniBatch) = lₛloss(model, xy.x, xy.path_cost, xy.H₊, xy.H₋)
lₛloss(model, mb::NamedTuple{(:minibatch,:stats)}) = lₛloss(model, mb.minibatch)

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

"""
descendants(sol, parents_id)

returns `descendants` of `parents_id` expanded during the search,
which are disjoint of `parents_id`, i.e.  
parents_id ∩ descendants(sol, parents_id) = ∅
"""
descendants(sol, parents_id::Vector) = descendants(sol, Set(parents_id))

function descendants(sol, parents_id::Set)
	childs = [s.id for s in values(sol.search_tree) if s.parent_id ∈ parents_id]
	setdiff(childs, parents_id)
end

nonempty(s::LₛMiniBatch) = !isempty(s.H₊) && !isempty(s.H₋)
nonempty(s::LgbfsMiniBatch) = !isempty(s.H₊) && !isempty(s.H₋)
nonempty(s::L₂MiniBatch)  = true
nonempty(s::NamedTuple{(:minibatch, :stats)}) = nonempty(s.minibatch)

function getloss(name)
	name == "l2" && return((L₂Loss(), L₂MiniBatch))
	name == "l₂" && return((L₂Loss(), L₂MiniBatch))
	name == "lstar" && return((LₛLoss(), LₛMiniBatch))
	name == "lₛ" && return((LgbfsLoss(), LgbfsMiniBatch))
	name == "lgbfs" && return((LgbfsLoss(), LgbfsMiniBatch))
	error("unknown loss $(name)")
end