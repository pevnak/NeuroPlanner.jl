######
# define a NN based solver
######
struct NeuroHeuristic{P,M} <: Heuristic 
	pddle::P
	model::M
end

NeuroHeuristic(pddld, problem, model) = NeuroHeuristic(NeuroPlanner.add_goalstate(pddld, problem), model)
Base.hash(g::NeuroHeuristic, h::UInt) = hash(g.model, hash(g.pddle, h))
SymbolicPlanners.compute(h::NeuroHeuristic, domain::Domain, state::State, spec::Specification) = only(h.model(h.pddle(state)))
