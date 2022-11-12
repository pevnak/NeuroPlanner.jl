using PDDL2Graph
using PDDL
using Flux
using GraphSignals
using GeometricFlux
using SymbolicPlanners
using Statistics
using IterTools

function create_example(pddld, domain, problem::GenericProblem)
	pddle = PDDL2Graph.add_goalstate(pddld, domain, problem)
	state = initstate(domain, problem)
	goal = PDDL.get_goal(problem)
	planner = AStarPlanner(HAdd())
	sol = planner(domain, state, goal)
	satisfy(domain, sol.trajectory[end], goal) || error("failed to solve the problem")
	x = reduce(cat, map(pddle, sol.trajectory));
	y = collect(length(sol.trajectory):-1:1);
	(;x,y)
end

function create_example(pddld, domain, filename::String)
	problem = load_problem(filename)
	create_example(pddld, domain, problem)	
end

function create_examples(pddld, domain, dirname::String)
	problem_files = filter(startswith("task"), readdir(dirname))
	map(f -> create_example(pddld, domain, joinpath(dirname, f)), problem_files)
end


benchdir(s...) = joinpath("benchmarks","blocks-slaney",s...)
domain = load_domain(benchdir("domain.pddl"))
pddld = PDDLExtractor(domain)

trn_dataset = create_examples(pddld, domain, benchdir("blocks3"))
tst_dataset = create_examples(pddld, domain, benchdir("blocks4"))

#crate model from some problem instance
model = let 
	problem = load_problem(benchdir("blocks3", "task01.pddl"))
	pddle = PDDL2Graph.add_goalstate(pddld, domain, problem)
	state = initstate(domain, problem)
	h₀ = pddle(state)
	MultiModel(h₀, 4, d -> Chain(Dense(d, 32,relu), Dense(32,1)))
end

loss(x, y) = Flux.Losses.mse(vec(model(x)), y)
loss(xy::NamedTuple{(:x,:y)}) = loss(xy.x,xy.y)
for i in 1:10
	Flux.train!(loss, Flux.params(model), repeatedly(() -> rand(trn_dataset), 1000), Flux.Optimise.AdaBelief())
	@show mean(loss(xy) for xy in trn_dataset)
	@show mean(loss(xy) for xy in tst_dataset)
end