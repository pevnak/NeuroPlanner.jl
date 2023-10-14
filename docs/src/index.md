# NeuroPlanner.jl

## Motivation
This library is an experimental library that implements:
1. neural network based heuristic functions that can accept problem and domains encoded in "unspecified" but sufficiently general subset of PDDL language.
2. various loss functions used to optimize parameters of the heuristic functions.

The library is in development and therefore things can change.

\[OUTDATED\] An api of heuristic function api and few details and gotchas can be found [here](heuristic.md).

## Usage example

A naive example of learning a heuristic on a domain

First we load the domain and problem files
```julia
using PDDL
using Neuroplanner
domain = load_domain("../domains/ferry/domain.pddl")
problem_files = [joinpath("../domains/ferry/", f) for f in readdir("../domains/ferry") if endswith(f,".pddl") && f !== "domain.pddl"]
train_files = filter(s -> isfile(plan_file(s)), problem_files)
problem = load_problem(first(problem_files))
```

Then we created an unspecialized extractor `pddld` for our chosen domain "ferry", here we will pick the HyperExtractor extractor.
```julia
pddld = HyperExtractor(domain)
```
The extractor is then specialised for the given problem, this is done by the `initproblem()` function which also return the initial state of the problem. With the specialised extractor (`pddle`) we can then create the extracted inital state `h₀`.  
```julia
pddle, state = initproblem(pddld, problem)

h₀ = pddle(state)
```
Now we can create a model to represent the heuristic. 
```julia
using Mill
using Flux
using SymbolicPlanners
using Logging
using Setfield
include("utils.jl")
model = reflectinmodel(h₀, d -> Dense(d, 32, relu);fsm = Dict("" =>  d -> ffnn(d, 32, 1, 2)))

opt = AdaBelief()
ps = Flux.params(model)
```
To allow for efficient training, we create minibatches from the training files. "lstar" is the name of the loss function we want to use.
```julia
include("problems.jl")
fminibatch = NeuroPlanner.minibatchconstructor("lstar") 
minibatches = map(train_files) do problem_file
			plan = load_plan(problem_file)
			problem = load_problem(problem_file)
			ds = fminibatch(pddld, domain, problem, plan)
		end
```
Finally, we train the model and run tests before and after to compare performance.
```julia
using SymbolicPlanners

planner = AStarPlanner

function solve_problem(pddld, problem::GenericProblem, model, init_planner; max_time=30, return_unsolved = false)
	domain = pddld.domain
	state₀ = initstate(domain, problem)
	hfun = NeuroHeuristic(pddld, problem, model)
	planner = init_planner(hfun; max_time, save_search = true)
	solution_time = @elapsed sol = planner(domain, state₀, PDDL.get_goal(problem))
	return_unsolved || sol.status == :success || return(nothing)
	stats = (;solution_time, 
		sol_length = length(sol.trajectory),
		expanded = sol.expanded,
		solved = sol.status == :success,
		time_in_heuristic = hfun.t[]
		)
	(;sol, stats)
end
solve_problem(pddld, problem_file::AbstractString, model, init_planner; kwargs...) = solve_problem(pddld, load_problem(problem_file), model, init_planner; kwargs...)

max_steps = 10000
include("training.jl")
train!(NeuroPlanner.loss, model, ps, opt, () -> rand(minibatches), max_steps; trn_data = minibatches)

```

```@docs
initproblem(ex, problem; add_goal = true)
```