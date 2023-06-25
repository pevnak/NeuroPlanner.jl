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

The parameters of heuristic function are implemented in a neural network, `nn`, which expects some sample as an input. This same is not directly a state in pddl, but an image if an  *extraction function* `ex`, which prepares the state `s`. This construction is reminiscent of **Mill.jl**, on top of which Lifted Relational NNs are built. The advantage is that the construction contains tedious non-differentiable operations and allows for minibatching. The heuristic function can be seen as a composition `nn ∘ ex.`


Let's now focus on extraction function `ex,` which takes a state `s` and converts it to some representation suitable for NN`. While in most cases, this representation would be a tensor, here we preder a relational encoding similar to Lifted Relational Neural Networks (LRNN). In this library, LRNN are instance of `KnowledgeBase`. The extraction function `ex` therefore to  controls the computational graph used by the heuristic function, including things like the number of graph convolutions. The  extraction function is implemented as a callable struct with a following api, where `HyperExtractor` is used as an example:
* `ex = HyperExtractor(domain)` --- initialize the extractor for a given domain
* `ex = specialize(ex, problem)`  --- specialize the extractor functions for a given domain
* `ex = add_goalstate(ex, problem, goal = goalstate(domain, problem)` --- fixes a goal state in the extractor
* `ex = add_initstate(ex, problem, start = initstate(domain, problem)` --- fixes an initial state in the extractor.

With this, `ex(state)` converts the state to a structure for the neural network, an instance of `KnowledgeBase` in this concerete example.

All extraction functions has to be initialized for a given domain and specialized for a given problem. This is typically needed to initiate various mapping, for example to ensure that same objects are mapped to same vertices in hypergraphs, etc. Adding goal or init state is optional. If they are added, the input to the neural network would always contain *goal* or *init* state, in which case the neural network will measure a distance to a state. If they are not used, the neural network can be used to create and embedding of states. 


### Example
Load the libraries, domain, and problem
```julia
using NeuroPlanner
using PDDL
using Flux
using Mill
using SymbolicPlanners
using PlanningDomains

domain = load_domain(IPCInstancesRepo,"ipc-2014", "barman-sequential-satisficing")
problems = list_problems(IPCInstancesRepo, "ipc-2014", "barman-sequential-satisficing")
problem = load_problem(IPCInstancesRepo, "ipc-2014", "barman-sequential-satisficing", first(problems))
```

First, we create the `HyperExtractor` for the `domain`
```julia
julia> ex = HyperExtractor(domain)
Unspecialized extractor for barman (6, 9)
```

Then, we specialize the extractor for a problem
```
julia> specex = specialize(ex, problem)
Specialized extractor without goal for barman (6, 9, 40)
```

If we would like to use the extractor to measure a distance to the goal, we add the goal
```julia
julia> specex = add_goalstate(specex, problem,)
Specialized extractor with goal for barman (6, 9, 40)
```
The function `add_goalstate` has a third implicit parameter `goal =  goalstate(domain, problem),` which allows to specify different goal then the default for the problem. Also, the function checks, if the extraction function is specialized for the problem and if not, it specialize it. Hence, the above can be shorted as 
```julia
julia> specex = add_goalstate(HyperExtractor(domain), problem)
Specialized extractor with goal for barman (6, 9, 40)
```


With specialized extraction function, we can convert a state to a `KnowledgeBase` as 
```julia
julia> s = initstate(domain, problem);
julia> specex(s)
KnowledgeBase: (x1,gnn_2,res_3,gnn_4,res_5,o)
```

The neural network processing the `KnowledgeBase` can be initialized as the neural network in **Mill.jl** library through `reflectinmodel` function
```julia
julia> kb = specex(s);
julia> model = reflectinmodel(specex(s), d -> Dense(d,10), SegmentedMean;fsm = Dict("" =>  d -> Dense(d,1)))
KnowledgeModel: (gnn_2,res_3,gnn_4,res_5,o)
```
which finishes the construction of the heuristic function as 
```julia
julia> model(specex(s))
1×1 Matrix{Float32}:
 0.003934524
```
The parameters of the model can be optimized using the standard method of **Flux.jl** on top of which they are builded.

A complete example should look like:
```julia
using NeuroPlanner
using PDDL
using Flux
using Mill
using SymbolicPlanners
using PlanningDomains

domain = load_domain(IPCInstancesRepo,"ipc-2014", "barman-sequential-satisficing")
problems = list_problems(IPCInstancesRepo, "ipc-2014", "barman-sequential-satisficing")
problem = load_problem(IPCInstancesRepo, "ipc-2014", "barman-sequential-satisficing", first(problems))

ex = HyperExtractor(domain)
specex = specialize(ex, problem)
specex = add_goalstate(ex, problem, goalstate(domain, problem))
s = initstate(domain, problem)
model = reflectinmodel(specex(s), d -> Dense(d,10), SegmentedMean;fsm = Dict("" =>  d -> Dense(d,1)))

model(specex(s))
```

### First remark: the model is general 

The model is able to process any problem instance despite it has been constructed from a state on a given problem instance. This can be seen on the following example which assumes the above model and uses the model on all problem instances from *barman*. Notice that the extractor needs to be specialized for every problem instance.
```julia
ex = HyperExtractor(domain)

map(problems) do problem_name
	problem = load_problem(IPCInstancesRepo, "ipc-2014", "barman-sequential-satisficing", problem_name)
	s = initstate(domain, problem)
	specex = add_goalstate(ex, problem)
	only(model(specex(s)))
end
```

### Second remark: fixing initial state measures distance from initial state
If the extractor is specialized with the goalstate, it meaures a distances from the state to a goalstate. On the other hand if the extractor is specialized with the init state, it will measure distance from init state to a state. Hence a distance from init to goal state can be computed by both specializations as is shown in the following example
```julia
ex = HyperExtractor(domain)
problem = load_problem(IPCInstancesRepo, "ipc-2014", "barman-sequential-satisficing", first(problems))

iex = add_initstate(ex, problem)
gex = add_goalstate(ex, problem)

si = initstate(domain, problem)
gi = goalstate(domain, problem)
model = reflectinmodel(iex(si), d -> Dense(d,10), SegmentedMean;fsm = Dict("" =>  d -> Dense(d,1)))

model(iex(goalstate(domain, problem))) ≈ model(gex(initstate(domain, problem)))
```


### Third remark: extractor without goal is useful for creating an embedding
May-be, we do not want a neural network to implement a heuristic function, but to project the state to a vector. This can be done with a specialized extractor without goal as 
```julia
problem = load_problem(IPCInstancesRepo, "ipc-2014", "barman-sequential-satisficing", first(problems))
ex = specialize(HyperExtractor(domain), problem)

si = initstate(domain, problem)
gi = goalstate(domain, problem)
model = reflectinmodel(ex(si), d -> Dense(d,10), SegmentedMean;fsm = Dict("" =>  d -> Dense(d,3)))
```
now the model will project states to the `17`-dimensional vector as
```julia
julia> model(ex(si))
3×1 Matrix{Float32}:
  0.048694983
 -0.35071477
 -0.013481511

```

### Fourth remark: extracted states can be batched
`KnowledgeBase` supports `Flux.batch` for minibatching. Using the above example, we can create a minibatch containing initial and goal state as
```julia
julia> b = Flux.batch([ex(si), ex(gi)])
KnowledgeBase: (x1,gnn_2,res_3,gnn_4,res_5,o)
```
and project it with the model
```julia
julia> model(b)
3×2 Matrix{Float32}:
  0.048695    0.061797
 -0.350715    0.111813
 -0.0134815  -0.0315447
```


## A complete example with an integration to the planner
```julia

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