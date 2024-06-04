# Usage guide
The Neuroplanner library allows for creation and training of a heuristic function to use in planning problems. This function is represented by a neural network model; therfore, for it to be ready for use, several steps must be completed.

---
## Model creation

To begin, necessarry libraries are imported.

```julia
using NeuroPlanner
using PDDL
using Flux
using JSON
using SymbolicPlanners
using PDDL: GenericProblem
using SymbolicPlanners: PathSearchSolution
using Statistics
using IterTools
using Random
using StatsBase
using Serialization
using DataFrames
using Mill
using Functors
using Accessors
using Setfield
using Logging
using TensorBoardLogger
```

Then a representation of the domain we will be working with is needed. PDDL has a helper function load_domain() into which the string path locationof domain.pddl file to be loaded is passed. 

```julia
domain = load_domain(domain_pddl)
```
Is also helpful to keep the string paths to the problem files we will be dealing with. They can be loaded for example as such:

```julia
problem_files = [joinpath("../domains/ferry/", f) for f in readdir("../domains/ferry") if endswith(f,".pddl") && f !== "domain.pddl"]
```
Neuroplanner has a function for loading both the domain and the problem files, which works with the provided (../files/domains.zip).

```julia
domain_pddl, problem_files = getproblem(domain_name)
```

---


Problems from the domain require an extractor (here called `pddld`) to be parsed in a standardised form. When creating an extractor we can choose from diferrent architectures (`ASNet`, `HyperExtractor`, `HGNNLite`, `HGNN`, `LevinASNet`), which fundamentally change how the problem, and as a result the model, are internally represented and will affect how the final heuristic behaves. Details here:
[Extractors](extractors.md), [Theoretical background](theory.md).

```julia
pddld = ASNet(domain)
pddld = HyperExtractor(domain)
pddld = HGNNLite(domain)
pddld = HGNN(domain)
pddld = LevinASNet(domain)
```

Now the chosen extractor has to be specialized. First a problem is loaded with the `load_problem()` function (any problem from the set will do, we are choosing the first one for simplicity). Next `specialize()` is used to create the specialized extractor `pddle`, and create the initial state of the problem with `initstate(domain, problem)`

```julia 
problem = load_problem(first(problem_files))
pddle = specialize(pddld, problem)
state = initstate(domain, problem)
```

Alternatively both specialization of the extractor and creation of the inital state can be done by calling `initproblem()`

```julia
pddle, state = initproblem(pddld, problem)
```

To use the specialized extractor, its functor is called on the state to be extracted. The inital state is extracted so the model can be created from it later.
```julia
h₀ = pddle(state)
```

---


With the extracted state the model is created with the Mill.jl `reflectinmodel()` function. This is an example of how the model creation can look. Brief explanation of the args used here:

`h₀` A state that the model will be able to process, specifies structure of the model.

`d -> Dense(d, dense_dim, relu)` Setting of dense layers, output will be = dense_dim, activation function is relu.

`fsm = Dict("" =>  d -> ffnn(d, dense_dim, 1, dense_layers))` optional kwarg, overrides constructions of feed-forward models. Here `""` in fsm denotes the last (output) layer, so these are settings for the last layer. (The 1 is the output dimension)

```julia
reflectinmodel(h₀, d -> Dense(d, dense_dim, relu);fsm = Dict("" =>  d -> ffnn(d, dense_dim, 1, dense_layers)))
```

For more optional keyword arguments and info on this function see [Mill.jl documentation on the topic](https://ctuavastlab.github.io/Mill.jl/stable/manual/reflectin/).

---
## Model training

To make training efficient the data is split into minibatches. In NeuroPlanner minibatches are tied one-to-one with loss functions; each loss function has a minibatch constructor that prepares the data in a way that the loss function can parse. Options for loss functions are: `l2`, `l₂`, `lstar`, `lₛ`, `lgbfs`, `lrt`, `bellman`, `levinloss`. With any of these the constructor can be created. For info on differences in loss functions see [Losses](losses.md), or the [Theoretical background](theory.md).

```julia
fminibatch = NeuroPlanner.minibatchconstructor("l2") 
fminibatch = NeuroPlanner.minibatchconstructor("l₂") 
fminibatch = NeuroPlanner.minibatchconstructor("lstar") 
fminibatch = NeuroPlanner.minibatchconstructor("lₛ") 
fminibatch = NeuroPlanner.minibatchconstructor("lgbfs") 
fminibatch = NeuroPlanner.minibatchconstructor("lrt") 
fminibatch = NeuroPlanner.minibatchconstructor("bellman") 
fminibatch = NeuroPlanner.minibatchconstructor("levinloss") 
```

With the constructor made, the batches can be created with the code snippet below. Deduplicate is a Neuroplanner function which eliminates redudant data from the batches, cutting down the size dramatically.

```julia
minibatches = map(problem_files) do problem_file
			  plan = load_plan(problem_file)
		 	  problem = load_problem(problem_file)
			  ds = fminibatch(pddld, domain, problem, plan)
			  dedu = @set ds.x = deduplicate(ds.x)
			  end
```

An [optimiser](https://fluxml.ai/Optimisers.jl/dev/api/#Optimisation-Rules) is chosen (any works, for brevity `AdaBelief()` is picked here) and model parameters are extracted.

```julia
opt = AdaBelief();
ps = Flux.params(model);
```

---


The model is trained with `train!()`. 

NeuroPlanner.loss passes the generic NeuroPlanner loss, which will dispatch to the correct loss based on the minibatch passed to it.

Model,ps,opt are the pre-prepared structs.  

() -> rand(minibatches) creates an anonymous function which returns a random minibatch (train! requires a function that returns batches).

max_steps is the maximum amount of steps the training will take.

```julia
train!(NeuroPlanner.loss, model, ps, opt, () -> rand(minibatches), max_steps)
```

With this the model is trained and ready to be used.

---
## Model usage

The simplest way of using the trained model is to call its functor. It computes the value of the heuristic function for the extracted state passed to it.

```julia
heur_value = model(pddle(state))
```

This, however, can be slow and unwieldy to work with at a large scale. NeuroPlanner provides several helper functions to make working with a trained model easier.

`solve_problem()` solves the problem in the problem_files, using the passed model, extractor and planner. It returns a solution object, which has fields describing details of how the problem was solved.

```julia
sol = solve_problem(pddld, problem_file, model, planner; return_unsolved = true)
```

If we want to compare to a non-heuristic solution, the `solveproblem()` solves the problem using a standard forward planner with a null heuristic. It only takes the domain of the problem and the problem file`` as the input.

```julia
sol = solveproblem(domain, problem_file)
```

It is also an option to create a `NeuroHeuristic()` which behaves like a normal heuristic object and can be passed to constructors of planners.

```julia
hfun = NeuroHeuristic(pddld, problem, model)
planner = AStarPlanner(hfun)
```

The planners availiable are [`AStarPlanner`, `GreedyPlanner`, `W15AStarPlanner`, `W20AStarPlanner`] and `BFSPlanner` which is used with the LevinAsnet model. Once created, the planner can be used to also get the solution of a given problem.

```julia
sol = planner(domain, state₀, PDDL.get_goal(problem))
```

To quickly display a collection of solutions, the `show_stats` which prints relevant data to console.

```julia
show_stats(solutions)
```