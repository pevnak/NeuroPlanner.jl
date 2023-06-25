# NeuroPlanner

## Motivation

This library is an experimental library that implements:
1. neural network based heuristic functions that can accept problem and domains encoded in "unspecified" but sufficiently general subset of PDDL language.
2. various loss functions used to optimize parameters of the heuristic functions.

The library is in development and therefore things can change.

## Heuristic functions
At the moment, we implement three different heuristic functions
* `HyperExtractor` which encodes state as relations coded in hypergraph (more on this later);
* `HGNN` is an encoding proposed in *Learning Domain-Independent Planning Heuristics with Hypergraph Networks, William Shen, Felipe Trevizan, Sylvie Thiebaux, 2020*
* `ASNet` is an encoding proposed in *Action Schema Networks: Generalised Policies with Deep Learning, 2018*

The heuristic function itself is implemented as a neural network that expects a sample of some structure. This structure is created by an *extraction function*, which takes a state and converts it to something that the neural network can process (in classical schemes, this would be tensor, here in this library, this is an instance of `KnowledgeBase`.) The extraction function therefore to a large extent controls the computational graph used by the heuristic function. The  extraction function is implemented as a callable struct with a following api. Let's use a `HyperExtractor` as an example
* `ex = HyperExtractor(domain)` --- initialize the extractor for a given domain
* `ex = specialize(ex, problem)`  --- specialize the extractor functions for a given domain
* `ex = add_goalstate(ex, problem, goal = goalstate(domain, problem)` --- fixes a goal state in the extractor
* `ex = add_initstate(ex, problem, start = initstate(domain, problem)` --- fixes an initial state in the extractor.

With this, `ex(state)` converts the state to a structure for the neural network.

All extraction functions has to be initialized for a given domain and specialed for a given problem. Adding goal or init state is optional. If they are added, the input to the neural network would always contain *goal* or *init* state, in which case the neural network will measure a distance to a state. If they are not used, the neural network can be used to create and embedding of states. 


The fixing of *init* and *goal* the state is there, because it is assumed



## Example
Is a naive attempt to learn heuristics over PDDL domains. It is restricted only to PDDL2 with predicates with arity at most 2, as it represents the state as a graph, where each object corresponds to one vertex and predicates with arity two are represented as edges. Very naive, but might work.

Below is a commented example which learns heuristic for a single instance of Sokoban. A complete example is located in `example.jl`.

Start as usually with few imports and load domain and problem instance.
```julia
using NeuroPlanner
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
pddle = NeuroPlanner.add_goalstate(pddld, problem)
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

GNNHeuristic(pddld, problem, model) = GNNHeuristic(NeuroPlanner.add_goalstate(pddld, problem), model)
Base.hash(g::GNNHeuristic, h::UInt) = hash(g.model, hash(g.pddle, h))
SymbolicPlanners.compute(h::GNNHeuristic, domain::Domain, state::State, spec::Specification) = only(h.model(h.pddle(state)))

problem = load_problem(filename)
pddle, state = initproblem(pddld, problem)
goal = PDDL.get_goal(problem)
planner = AStarPlanner(GNNHeuristic(pddld, problem, model))
sol = planner(domain, state, goal)
satisfy(domain, sol.trajectory[end], goal)
``` 