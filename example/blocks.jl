using PDDL2Graph
using PDDL
using Flux
using GraphSignals
using GeometricFlux
using SymbolicPlanners
using Statistics
using IterTools

benchdir(s...) = joinpath("benchmarks","blocks-slaney",s...)
taskfiles(s) = [benchdir(s, f) for f in readdir(benchdir(s)) if startswith(f,"task",)]

# define some examples 
function create_example(pddld, problem::GenericProblem)
	domain = pddld.domain
	pddle = PDDL2Graph.add_goalstate(pddld, problem)
	state = initstate(domain, problem)
	goal = PDDL.get_goal(problem)
	planner = AStarPlanner(HAdd())
	sol = planner(domain, state, goal)
	satisfy(domain, sol.trajectory[end], goal) || error("failed to solve the problem")
	x = reduce(cat, map(pddle, sol.trajectory));
	y = collect(length(sol.trajectory):-1:1);
	(;x,y)
end

create_example(pddld, filename::String) = create_example(pddld, load_problem(filename))	
create_examples(pddld, dirname::String) = [create_example(pddld, f) for f in taskfiles(dirname)]

domain = load_domain(benchdir("domain.pddl"))
pddld = PDDLExtractor(domain)

dataset = create_examples(pddld, "blocks3")

#crate model from some problem instance
model = let 
	problem = load_problem(benchdir("blocks3", "task01.pddl"))
	pddle, state = initproblem(pddld, problem)
	h₀ = pddle(state)
	odim_of_graph_conv = 4
	MultiModel(h₀, odim_of_graph_conv, d -> Chain(Dense(d, 32,relu), Dense(32,1)))
end

loss(x, y) = Flux.Losses.mse(vec(model(x)), y)
loss(xy::NamedTuple{(:x,:y)}) = loss(xy.x,xy.y)
for i in 1:10
	Flux.train!(loss, Flux.params(model), repeatedly(() -> rand(dataset), 1000), Flux.Optimise.AdaBelief())
	@show mean(loss(xy) for xy in dataset)
end


# Let's make it a proper heuristic for SymbolicPlanners
struct GNNHeuristic{P,M} <: Heuristic 
	pddle::P
	model::M
end

GNNHeuristic(pddld, problem, model) = GNNHeuristic(PDDL2Graph.add_goalstate(pddld, problem), model)
Base.hash(g::GNNHeuristic, h::UInt) = hash(g.model, hash(g.pddle, h))
SymbolicPlanners.compute(h::GNNHeuristic, domain::Domain, state::State, spec::Specification) = only(h.model(h.pddle(state)))

map(taskfiles("blocks4")) do filename
	problem = load_problem(filename)
	pddle, state = initproblem(pddld, problem)
	goal = PDDL.get_goal(problem)
	planner = AStarPlanner(GNNHeuristic(pddld, problem, model))
	sol = planner(domain, state, goal)
	satisfy(domain, sol.trajectory[end], goal)
end
