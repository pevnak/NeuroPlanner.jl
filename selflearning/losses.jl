#############
#	L2 Losses
#############
struct L₂MiniBatch{X,Y}
	x::X 
	y::Y
end

function L₂MiniBatch(sol, pddld, problem)
   pddle, state = NeuroPlanner.initproblem(pddld, problem)
   L₂MiniBatch(batch(map(pddle, sol.trajectory)),
     collect(length(sol.trajectory):-1:1),
     )
end

struct L₂loss end 
(l::L₂loss)(args...) = l₂loss(args...)
l₂loss(model, x, y) = Flux.Losses.mse(vec(model(x)), y)
l₂loss(model, xy::NamedTuple) = l₂loss(model, xy.x, xy.y)
l₂loss(model, xy::L₂MiniBatch) = l₂loss(model, xy.x, xy.y)
loss(model, xy::L₂MiniBatch) = l₂loss(model, xy.x, xy.y)


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

function LₛMiniBatch(sol, pddld, problem)
	sol.search_tree === nothing && error("solve the problem with `save_search=true` to keep the search tree")
	pddle, state = NeuroPlanner.initproblem(pddld, problem)

	# get indexes of the states on the solution path, which seems to be hashes of states 
	trajectory_id = hash.(sol.trajectory)
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

(l::LₛLoss)(args...) = lₛloss(args...)

function lₛloss(model, x, g, H₊, H₋)
	g = reshape(g, 1, :)
	f = model(x) .+ g
	o = f * H₋ .- f * H₊
	isempty(o) && return(zero(eltype(o)))
	mean(softplus.(o))
end
lₛloss(model, xy::LₛMiniBatch) = lₛloss(model, xy.x, xy.path_cost, xy.H₊, xy.H₋)

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

function LgbfsMiniBatch(sol, pddld, problem)
	l = LₛMiniBatch(sol, pddld, problem)
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