# NeuroPlanner.jl

## Motivation
This library implements two things:
1. *heuristic functions* for STRIPS domains implemented by (hyper-multi-) graph neural networks; 
2. loss functions to optimize parameters of the heuristic functions from a solved plans.

The library is experimental in the sense that it is used to research different phenomenons and trade-offs.  This also means that things can change.

\[OUTDATED\] An api of heuristic function api and few details and gotchas can be found [here](heuristic.md).

## Short snipper

```julia
using NeuroPlanner
using NeuroPlanner.PDDL
using NeuroPlanner.Mill
using NeuroPlanner.Flux
using NeuroPlanner.SymbolicPlanners
using NeuroPlanner.Flux.Optimisers
domain = load_domain("../domains/ferry/domain.pddl")
problem_files = [joinpath("../domains/ferry/", f) for f in readdir("../domains/ferry") if endswith(f,".pddl") && f !== "domain.pddl"]
train_files = filter(s -> isfile(plan_file(s)), problem_files)
problem = load_problem(first(problem_files))

pddld = ObjectAtom(domain)
pddle, state = initproblem(pddld, problem)
h₀ = pddle(state)
model = reflectinmodel(h₀, d -> Dense(d, 32, relu);fsm = Dict("" =>  d -> Chain(Dense(d, 32, relu), Dense(32,1))))

fminibatch = NeuroPlanner.minibatchconstructor("lstar") 
minibatches = map(train_files) do problem_file
			plan = NeuroPlanner.load_plan(problem_file)
			problem = load_problem(problem_file)
			ds = fminibatch(pddld, domain, problem, plan)
		end

state_tree = Optimisers.setup(Optimisers.AdaBelief(), model) 
for i in 1:10_000
	mb = rand(minibatches)
	l, ∇model = Flux.withgradient(model -> NeuroPlanner.loss(model, mb), model)
	state_tree, model = Optimisers.update(state_tree, model, ∇model[1]);
end
```


## A more detailed walkthrough

A naive example of learning a heuristic on a domain

First we load the domain and problem files
```julia
using NeuroPlanner
using NeuroPlanner.PDDL
domain = load_domain("../domains/ferry/domain.pddl")
problem_files = [joinpath("../domains/ferry/", f) for f in readdir("../domains/ferry") if endswith(f,".pddl") && f !== "domain.pddl"]
train_files = filter(s -> isfile(plan_file(s)), problem_files)
problem = load_problem(first(problem_files))
```

Then we create an extractor `pddld` for the chosen domain "ferry", here we will pick the `ObjectAtom` extractor. The role of the extracor is to convert the STRIPS state to a representation suitable for the neural networks. The specialization for the domain fixes information which are constant 
```julia
pddld = ObjectAtom(domain)
```
The extractor is then specialised for the given problem, this is done by the `initproblem()` function which also return the initial state of the problem. With the specialised extractor (`pddle`) we can then create the extracted inital state `h₀`.  
```julia
pddle, state = initproblem(pddld, problem)
h₀ = pddle(state)
```
Now we can create a model to represent the heuristic. 
```julia
using NeuroPlanner.Mill
using NeuroPlanner.Flux
using NeuroPlanner.SymbolicPlanners
model = reflectinmodel(h₀, d -> Dense(d, 32, relu);fsm = Dict("" =>  d -> Chain(Dense(d, 32, relu), Dense(32,1))))
```
To allow for efficient training, we create minibatches from the training files. Each loss function has its own type of minibatch, which allows a dispatch of the loss on the type of the minibatch. In below example, the function `fminibatch` therefore creates minibatch for loss function named "lstar", which is designed to maximize efficiency of A* algorithm [^1].
```julia
fminibatch = NeuroPlanner.minibatchconstructor("lstar") 
minibatches = map(train_files) do problem_file
			plan = NeuroPlanner.load_plan(problem_file)
			problem = load_problem(problem_file)
			ds = fminibatch(pddld, domain, problem, plan)
		end
```

With minibatches, we train the model using standart training loop.
```julia
using NeuroPlanner.Flux.Optimisers
state_tree = Optimisers.setup(Optimisers.AdaBelief(), model) 
for i in 1:10_000
	mb = rand(minibatches)
	l, ∇model = Flux.withgradient(model -> NeuroPlanner.loss(model, mb), model)
	state_tree, model = Optimisers.update(state_tree, model, ∇model[1]);
end
```

Finally, we evaluate the model inside the A* algorith,=m.
```julia
map(s -> NeuroPlanner.solve_problem(pddld, load_problem(s), model, AStarPlanner), setdiff(problem_files, train_files))
```

```@docs
initproblem(ex, problem; add_goal = true)
```

[^1]: Chrestien, Leah, et al. "Optimize Planning Heuristics to Rank, not to Estimate Cost-to-Goal." Advances in Neural Information Processing Systems 36 (2024).