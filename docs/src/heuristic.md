
## Heuristic functions
The heuristic function is composed of two parts. First, called **extractor**, `ex` takes a STRIPS state `s` and project it to to (computation) graph (more on this later). The second is the neural network `nn`, which takes the computation graph and project it to the heuristic function. This functionality is based on a custom extension of [Mill.jl](https://github.com/CTUAvastLab/Mill.jl) library. 
The complete heuristic function is therefore composition `nn ∘ ex`. 

The advantage is that non-differentiable operations are kept in the the extractor part and differentiable in the neural network. The main advantage of the extractor producing the computational graph is that it can be then deduplicated, which speeds-up the traingn.




### Extractors
The library implements various methods how to represent STRIPS states as graphs. This representation is important for the properties of the heuristic function, mainly to its ability to discriminate between states. This representation is perpendicular to the type of graph neural networks, in which this library in not that interested that much. The functionality projecting state to graph is called **extractor**.

The available extractors are:
* `ObjectAtom`[^1] represent state as a hyper-multi-graph. Each vertex corresponds to the object, each atom represent a hyper-edge. Different types of atoms are represented as different type of edges.
* `ObjectAtomBip`[^1] represent state as a multi-graph or graphs with features on edges. Each object and atom corresponds to a vertex. Object-vertex is connected to atom-vertex when object is an argument of the atom. The representation is similar to the `ObjectAtom`, except the hyper-edges are represented in bipartite graph.
* `ObjectBinary`[^1] represent states as multi graph (or graph with features on edges). Each object correponds to the vertex. Vertices are connected by the edge if they are in the same atom. The type of edge (or features one edges) corresponds to the type of atom and position of the object in the argument.
* `AtomBinary`[^1] represent states as multi graph (or graph with features on edges). Each object correponds to the atom. Vertices are connected by the edge if they share the same object. The type of the edge (or features one edges) corresponds to the position of the object in both atoms.
* `ObjectPair`[^1] each vertex corresponds to a tuple of objects and edges are create by some cryptic algorithm.
* `ASNet`[^2] creates vertices for each possible atoms. The atoms are present in the graph even when they are not `true` in the state. This means that graph representing states differ only in features on edges, which codes if the atom is `true` or `false.` 
* `HGNN`[^3] is similar to `ASNet`, except the message-passing over the hyper-edges is a bit different, as it includes more domain knowledge from the planning community.

Let's now focus on extraction function `ex`, which takes a state `s` and converts it to some representation suitable for NN. In the case of this library, the representation is an instance of `KnowledgeBase`, which encodes the copmutation graph. Since the extractor produces the compuation graph, the extraction function controls the number of graph convolutions and the presence of residual connections. 

The extraction function is implemented as a callable struct with a following api, where `ObjectBinary` is used as an example:
The api / interface of the extraction function is as follows:
* `ex = ObjectBinary(domain; message_passes = 2, residual=:linear, edgebuilder = FeaturedEdgeBuilder)` --- initialize the extractor for a given domain. At this moment, we need to specify the number of message passes and the type of residula layer (`:none` or `:linear`). Additionally, you specify how to represent edges pf different types by passing the `edgebuilder`. The default  `FeaturedEdgeBuilder`, uses is edges with features, other option is `MultiEdgeBuilder`  which uses multi-graph (multiple)
* `ex = specialize(ex, problem)`  --- specialize the extractor functions for a given domain
* `ex = add_goalstate(ex, problem, goal = goalstate(domain, problem)` --- fixes a goal state in the extractor
* `ex = add_initstate(ex, problem, start = initstate(domain, problem)` --- fixes an initial state in the extractor.

With this, `ex(state)` converts the state to a structure for the neural network, an instance of `KnowledgeBase` in this concerete example.

All extraction functions has to be initialized for a given domain and specialized for a given problem. This is typically needed to initiate various mapping to ensure it does not change between problem instances (an example is a map of categorical variables to one-hot representations).  Adding goal or init state is optional. If they are added, the input to the neural network would always contain *goal* or *init* state, in which case the neural network will measure a distance to a state. If they are not used, the neural network can be used to create and embedding of states. 


### Example
Load the libraries, domain, and problem
```julia
using NeuroPlanner
using NeuroPlanner.PDDL
using NeuroPlanner.Flux
using NeuroPlanner.Mill
using NeuroPlanner.SymbolicPlanners
using PlanningDomains

domain = load_domain(IPCInstancesRepo,"ipc-2014", "barman-sequential-satisficing")
problems = list_problems(IPCInstancesRepo, "ipc-2014", "barman-sequential-satisficing")
problem = load_problem(IPCInstancesRepo, "ipc-2014", "barman-sequential-satisficing", first(problems))
```

First, we create the `ObjectBinary` for the `domain`
```julia
julia> ex = ObjectBinary(domain)
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
julia> specex = add_goalstate(ObjectAtom(domain), problem)
Specialized extractor with goal for barman (6, 9, 40)
```


With specialized extraction function, we can convert a state to a `KnowledgeBase` as 
```julia
julia> s = initstate(domain, problem);
julia> specex(s)
KnowledgeBase: (x1,gnn_2,res_3,gnn_4,res_5,o)
```

The neural network processing the `KnowledgeBase` can be initialized as the neural network in [Mill.jl](https://github.com/CTUAvastLab/Mill.jl) library through extended `reflectinmodel` function
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
The parameters of the model can be optimized using the standard method of **Flux.jl** on top of which they are built. We refer the reader to the documentation of [Mill.jl](https://github.com/CTUAvastLab/Mill.jl) for details of the `reflectinmodel` function.

A complete example look like:
```julia
using NeuroPlanner
using NeuroPlanner.PDDL
using NeuroPlanner.Flux
using NeuroPlanner.Mill
using NeuroPlanner.SymbolicPlanners
using PlanningDomains

domain = load_domain(IPCInstancesRepo,"ipc-2014", "barman-sequential-satisficing")
problems = list_problems(IPCInstancesRepo, "ipc-2014", "barman-sequential-satisficing")
problem = load_problem(IPCInstancesRepo, "ipc-2014", "barman-sequential-satisficing", first(problems))

ex = ObjectAtom(domain)
specex = specialize(ex, problem)
specex = add_goalstate(ex, problem, goalstate(domain, problem))
s = initstate(domain, problem)
model = reflectinmodel(specex(s), d -> Dense(d,10), SegmentedMean;fsm = Dict("" =>  d -> Dense(d,1)))

model(specex(s))
```

## Remarks

### First remark: the model is general 

The model is able to process any problem instance despite it has been constructed from a state on a given problem instance. This can be seen on the following example which assumes the above model and uses the model on all problem instances from *barman*. Notice that the extractor needs to be specialized for every problem instance.
```julia
ex = ObjectAtom(domain)

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
ex = ObjectAtom(domain)
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
ex = specialize(ObjectAtom(domain), problem)

si = initstate(domain, problem)
gi = goalstate(domain, problem)
model = reflectinmodel(ex(si), d -> Dense(d,10), SegmentedMean;fsm = Dict("" =>  d -> Dense(d,3)))
```
now the model will project states to the `3`-dimensional vector as
```julia
julia> model(ex(si))
3×1 Matrix{Float32}:
  0.048694983
 -0.35071477
 -0.013481511

```
Notice the difference in the argument `fsm = Dict("" =>  d -> Dense(d,3))` in the argument of the `reflectinmodel`.

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

[^1]: Horčík, Rostislav, and Gustav Šír. "Expressiveness of Graph Neural Networks in Planning Domains." Proceedings of the International Conference on Automated Planning and Scheduling. Vol. 34. 2024.

[^2]: Toyer, Sam, et al. "Action schema networks: Generalised policies with deep learning." Proceedings of the AAAI Conference on Artificial Intelligence. Vol. 32. No. 1. 2018.

[^3]: Learning Domain-Independent Planning Heuristics with Hypergraph Networks, William Shen, Felipe Trevizan, Sylvie Thiebaux, 2020

