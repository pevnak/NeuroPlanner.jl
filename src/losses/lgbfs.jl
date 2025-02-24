
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

