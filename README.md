# PDDL2Graph
Is a naive attempt to learn heuristics over PDDL domains. It is restricted only to PDDL2 with predicates with arity at most 2, as it represents the state as a graph, where each object corresponds to one vertex and predicates with arity two are represented as edges. Very naive, but might work.

Below is a commented example which learns heuristic for a single instance of Sokoban. A complete example is located in `example.jl`.

Start as usually with few imports and load domain and problem instance.
```julia
using PDDL2Graph
using PDDL
using Flux
using GraphSignals
using GeometricFlux
using SymbolicPlanners

domain = load_domain("../test/sokoban.pddl")
problem = load_problem("../test/s1.pddl")
```

Then, we need to prepare extractor `pddld` for a domain . This extractor define order of predicates, which should not change, otherwise the neural network would be totally confused. An extractor for a domain has to be further specialized for a given problem instance to problem instance extractor `pddle`. This adds to the extractor a representation of the goal state and define order of vertices ensuring that goal state will be consistent with other states from this problem instance. 
```julia
pddld = PDDLExtractor(domain)
pddle = PDDL2Graph.add_goalstate(pddld, problem)
```

Than, we get an initial state, which allows us to define the model. The model is not much flexible. It contains two graph attention layers (you can specify their output dimension) and then you can specify the feed forward neural network processing the output after being aggergated. For simplicity with dimentions, you should provide a function constructing this dense part as in the below example.
```julia
state = initstate(domain, problem)
h₀ = pddle(state)
model = MultiModel(h₀, 4, d -> Chain(Dense(d, 32,relu), Dense(32,1)))
```

In this example, we construct training sample by running A* planner with `h_add` heuristic.

```julia
state = initstate(domain, problem)
goal = PDDL.get_goal(problem)
planner = AStarPlanner(HAdd())
sol = planner(domain, state, goal)
satisfy(domain, sol.trajectory[end], goal) || error("failed to solve the problem")
```

From the plan, we construct single minibatch with a cost-to-goal as a target. We emphasize that we allow to join multiple graphs to a minibatch using `reduce(cat, xx)`, but division is not supported. 
```julia
xx = map(pddle, sol.trajectory);
batch = reduce(cat, xx);
yy = collect(length(sol.trajectory):-1:1);
```

Finally, we can minimize the loss function using standard `Flux.train!`.
```julia
loss(args...) = Flux.Losses.mse(vec(model(batch)), yy)
println("loss before training: ", loss())
Flux.train!(loss, Flux.params(model), 1:1000, Flux.Optimise.AdaBelief())
println("loss after training: ", loss())
```

## Using trained model as a heuristic in solver of SymbolicPlanners
```julia
struct GNNHeuristic{P,M} <: Heuristic 
	pddle::P
	model::M
end

GNNHeuristic(pddld, problem, model) = GNNHeuristic(PDDL2Graph.add_goalstate(pddld, problem), model)
Base.hash(g::GNNHeuristic, h::UInt) = hash(g.model, hash(g.pddle, h))
SymbolicPlanners.compute(h::GNNHeuristic, domain::Domain, state::State, spec::Specification) = only(h.model(h.pddle(state)))

problem = load_problem(filename)
pddle, state = initproblem(pddld, problem)
goal = PDDL.get_goal(problem)
planner = AStarPlanner(GNNHeuristic(pddld, problem, model))
sol = planner(domain, state, goal)
satisfy(domain, sol.trajectory[end], goal)
``` 

## Simple and Sparse Graphs
When PDDL representation is converted to graph, the conversion routine uses on `SimpleGraph` from `Graphs.jl`, since it represents the graph as an adjacency list, which is easy for adding edges. The returned `MultiGraph` then contains tuple of `SimpleGraph`s, where there is one `SimpleGraph` for one type of edges. When graphs in `MultiGraph` are stored in `SimpleGraph`, the resulting type as an alias `SimpleMultiGraph` and crucially they can be concatenated to minibatch. Concatenation for `SimpleMultiGraph`s is easy due to the simplicity in manipulating adjacency list. Contrary, `GeometricFlux` understands `SparseGraphs`, where the adjacency matrix is stored as a `SparseMatrixCSC`, for which we did not implemented concatenation to minibatches. `MultiGNNLayer` automatically converts `SimpleMultiGraph`s to `SparseMultiGraph`s, therefore user does not even notice. you can perform the conversion by itself by method `simplegraph`. The suggested construction of one minibatch containing `minibatch_states` should therefore be
```julia
xx = pddle.(minibatch_states)
batch = reduce(cat, xx);
sbatch = PDDL2Graph.sparsegraph(batch);
```
*We emphasize that resulting `sbatch` cannot be concatenated with other `sbatch`, unlike `batch`.* Thus `sparsegraph` effectively freezes the representation. Using `SparseMultiGraph` put less stress on `Zygote`, but we have not observed huge difference in practice.