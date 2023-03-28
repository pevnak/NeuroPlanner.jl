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
using Setfield
using Comonicon
using TensorBoardLogger

include("solution_tracking.jl")
include("problems.jl")
include("training.jl")

function ffnn(idim, hdim, odim, nlayers)
	nlayers == 1 && return(Dense(idim,odim))
	nlayers == 2 && return(Chain(Dense(idim, hdim, relu), Dense(hdim,odim)))
	nlayers == 3 && return(Chain(Dense(idim, hdim, relu), Dense(hdim, hdim, relu), Dense(odim,odim)))
	error("nlayers should be only in [1,3]")
end

function experiment(domain_name, domain_pddl, train_files, problem_files, filename, fminibatch;max_steps = 10000, max_time = 30, graph_layers = 2, residual = true, dense_layers = 2, dense_dim = 32)
	!isdir(dirname(filename)) && mkpath(dirname(filename))
	logger=TBLogger(dirname(filename), prefix = basename(filename)[1:end-4])
	domain = load_domain(domain_pddl)
	# pddld = PDDLExtractor(domain)
	pddld = HyperExtractor(domain; message_passes = graph_layers, residual)

	#create model from some problem instance
	model, dedup_model = let 
		problem = load_problem(first(problem_files))
		pddle, state = initproblem(pddld, problem)
		h₀ = pddle(state)
		# model = reflectinmodel(h₀, d -> ffnn(d, dense_dim, dense_dim, dense_layers);fsm = Dict("" =>  d -> ffnn(d, dense_dim, 1, dense_layers)))
		model = reflectinmodel(h₀, d -> Dense(d, dense_dim, relu);fsm = Dict("" =>  d -> ffnn(d, dense_dim, 1, dense_layers)))
		dedup_model = reflectinmodel(h₀, d -> Dense(d,32) ;fsm = Dict("" =>  d -> Dense(d, 32)))
		model, dedup_model
	end

	minibatches = map(train_files) do problem_file
		plan = deserialize(plan_file(domain_name, problem_file))
		problem = load_problem(problem_file)
		ds = fminibatch(pddld, domain, problem, plan.plan)
		@set ds.x = deduplicate(dedup_model, ds.x)
	end

	# we can check that the model can learn the training data if the dedup_model can differentiate all input states, which is interesting by no means

	if isfile(filename[1:end-4]*"_model.jls")
		model = deserialize(filename[1:end-4]*"_model.jls")
	else
		opt = AdaBelief();
		ps = Flux.params(model);
		train!(NeuroPlanner.loss, model, ps, opt, () -> rand(minibatches), max_steps; logger, trn_data = minibatches)
		serialize(filename[1:end-4]*"_model.jls", model)	
	end

	stats = map(Iterators.product([AStarPlanner], problem_files)) do (planner, problem_file)
		used_in_train = problem_file ∈ train_files
		@show problem_file
		sol = solve_problem(pddld, problem_file, model, planner; return_unsolved = true)
		trajectory = sol.sol.status == :max_time ? nothing : sol.sol.trajectory
		merge(sol.stats, (;used_in_train, planner = "$(planner)", trajectory, problem_file))
	end
	df = DataFrame(vec(stats))
	mean(df.solved[.!df.used_in_train])
	serialize(filename, stats)
end

"""
ArgParse example implemented in Comonicon.

# Arguments

- `problem_name`: a name of the problem to solve ("ferry", "gripper", "blocks", "npuzzle")
- `loss_name`: 

# Options

- `--max_steps <Int>`: maximum number of steps of SGD algorithm (default 20_000)
- `--max_time <Int>`:  maximum steps of the planner used for evaluation (default 30)
- `--graph_layers <Int>`:  maximum number of layers of (H)GNN (default 1)
- `--dense_dim <Int>`:  dimension of all hidden layers, even those realizing graph convolutions (default  32)
- `--dense_layers <Int>`:  number of layers of dense network after pooling vertices (default 32)
- `--residual <String>`:  residual connections between graph convolutions (none / dense / linear)
"""

@main function main(domain_name, loss_name; max_steps::Int = 20_000, max_time::Int = 30, graph_layers::Int = 1, 
		dense_dim::Int = 32, dense_layers = 2, residual::String = "none", seed::Int = 1)
	Random.seed!(seed)
	residual = Symbol(residual)
	domain_pddl, problem_files, ofile = getproblem(domain_name, false)
	problem_files = filter(s -> isfile(plan_file(domain_name, s)), problem_files)
	train_files = filter(s -> isfile(plan_file(domain_name, s)), problem_files)
	train_files = sample(train_files, div(length(problem_files), 2), replace = false)
	fminibatch = NeuroPlanner.minibatchconstructor(loss_name)

	filename = joinpath("superhyper", domain_name, join([loss_name, max_steps,  max_time, graph_layers, residual, dense_layers, dense_dim, seed], "_")*".jls")
	experiment(domain_name, domain_pddl, train_files, problem_files, filename, fminibatch; max_steps, max_time, graph_layers, residual, dense_layers, dense_dim)
end