######
# define a NN based solver
######
struct NeuroHeuristic{P,M} <: Heuristic 
	pddle::P
	model::M
	t::Base.RefValue{Float64}
end

NeuroHeuristic(pddld, problem, model) = NeuroHeuristic(NeuroPlanner.add_goalstate(pddld, problem), model, Ref(0.0))
Base.hash(g::NeuroHeuristic, h::UInt) = hash(g.model, hash(g.pddle, h))
function SymbolicPlanners.compute(h::NeuroHeuristic, domain::Domain, state::State, spec::Specification) 
	h.t[] += @elapsed r = only(h.model(h.pddle(state)))
	r
end

using NeuroPlanner: LevinModel
function SymbolicPlanners.compute(h::NeuroHeuristic{<:LevinASNet, <:LevinModel}, domain::Domain, state::State, spec::Specification) 
	h.t[] += @elapsed r = h.model(h.pddle(state))
	(only(r[1]), vec(r[2]))
end
