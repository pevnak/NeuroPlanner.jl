using PDDL2Graph
using PDDL
using Graphs
using Flux
using Julog
using GraphSignals
using GeometricFlux
using SymbolicPlanners
# using GraphPlot
# using Cairo
# using Compose


domain = load_domain("sokoban.pddl")
problem = load_problem("s1.pddl")

# domain = load_domain("benchmarks/blocks-slaney/domain.pddl")
# problem = load_problem("benchmarks/blocks-slaney/blocks10/task01.pddl")

# domain = load_domain("benchmarks/ferry/ferry.pddl")
# problem = load_problem("benchmarks/ferry/train/ferry-l2-c1.pddl")

# domain = load_domain("benchmarks/gripper/domain.pddl")
# problem = load_problem("benchmarks/gripper/problems/gripper-n1.pddl")

# domain = load_domain("benchmarks/n-puzzle/domain.pddl")
# problem = load_problem("benchmarks/n-puzzle/train/n-puzzle-2x2-s1.pddl")

# domain = load_domain("benchmarks/zenotravel/domain.pddl")
# problem = load_problem("benchmarks/zenotravel/train/zenotravel-cities2-planes1-people2-1864.pddl")

pddle = PDDLExtractor(domain, problem) 
state = initstate(domain, problem)
gstate = goalstate(domain, problem)

h₀ = PDDL2Graph.multigraph(pddle, state)
m = MultiModel(h₀, 4, d -> Chain(Dense(d, 32,relu), Dense(32,1)))
ps = Flux.params(m)
gs = gradient(() -> sum(m(h₀)), ps)
[gs[p] for p in ps]

# get training example by running A* planner with h_add heuristic
state = initstate(domain, problem)
goal = PDDL.get_goal(problem)
planner = AStarPlanner(HAdd())
sol = planner(domain, state, goal)
plan = collect(sol)
trajectory = sol.trajectory
satisfy(domain, sol.trajectory[end], goal)

#construct training set for L2 loss
xx = [PDDL2Graph.multigraph(pddle, state) for state in sol.trajectory];
yy = collect(length(sol.trajectory):-1:1);

ps = Flux.params(m);
gs = gradient(ps) do 
	map(xx, yy) do h₀, y
		(sum(m(h₀)) - y)^2
	end |> sum 
end;
[gs[p] for p in ps]
