using PDDL2Graph
using PDDL
using Flux
using GraphSignals
using GeometricFlux
using SymbolicPlanners


domain = load_domain("../test/sokoban.pddl")
problem = load_problem("../test/s1.pddl")

# prepare extractor for the domain.
pddld = PDDLExtractor(domain)


#Get the extractor for the problem instance
pddle = PDDL2Graph.add_goalstate(pddld, domain, problem)
state = initstate(domain, problem)

# PDDL2Graph.multigraph
h₀ = pddle(state)
model = MultiModel(h₀, 4, d -> Chain(Dense(d, 32,relu), Dense(32,1)))

# get training sample by running A* planner with h_add heuristic
state = initstate(domain, problem)
goal = PDDL.get_goal(problem)
planner = AStarPlanner(HAdd())
sol = planner(domain, state, goal)
satisfy(domain, sol.trajectory[end], goal) || error("failed to solve the problem")

#construct a single minibatch with cost-to-goal as a target
xx = map(pddle, sol.trajectory);
batch = reduce(cat, xx);
yy = collect(length(sol.trajectory):-1:1);

loss(args...) = Flux.Losses.mse(vec(model(batch)), yy)
loss()
Flux.train!(loss, Flux.params(model), 1:1000, Flux.Optimise.AdaBelief())
loss()