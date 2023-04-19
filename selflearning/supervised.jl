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
using Logging
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

function tblogger(filename; min_level::LogLevel=Info, step_increment = 1)
	!isdir(dirname(filename)) && mkpath(dirname(filename))
    logdir = dirname(filename)

    evfile     = open(filename, "w")
    ev_0 = TensorBoardLogger.Event(wall_time=time(), step=0, file_version="brain.Event:2")
    TensorBoardLogger.write_event(evfile, ev_0)

    all_files  = Dict(filename => evfile)
    start_step = 0

    TBLogger{typeof(logdir), typeof(evfile)}(logdir, evfile, all_files, start_step, step_increment, min_level)
end

function experiment(domain_name, hnet, domain_pddl, train_files, problem_files, filename, fminibatch;max_steps = 10000, max_time = 30, graph_layers = 2, residual = true, dense_layers = 2, dense_dim = 32, settings = nothing)
	!isdir(dirname(filename)) && mkpath(dirname(filename))
	domain = load_domain(domain_pddl)
	pddld = hnet(domain; message_passes = graph_layers, residual)

	#create model from some problem instance

	# we can check that the model can learn the training data if the dedup_model can differentiate all input states, which is interesting by no means

	model = if isfile(filename*"_model.jls")
		deserialize(filename*"_model.jls")
	else
		model = let 
			problem = load_problem(first(train_files))
			pddle, state = initproblem(pddld, problem)
			h₀ = pddle(state)
			# model = reflectinmodel(h₀, d -> ffnn(d, dense_dim, dense_dim, dense_layers);fsm = Dict("" =>  d -> ffnn(d, dense_dim, 1, dense_layers)))
			reflectinmodel(h₀, d -> Dense(d, dense_dim, relu);fsm = Dict("" =>  d -> ffnn(d, dense_dim, 1, dense_layers)))
		end

		logger=tblogger(filename*"_events.pb")
		t = @elapsed minibatches = map(train_files) do problem_file
			plan = load_plan(problem_file)
			problem = load_problem(problem_file)
			ds = fminibatch(pddld, domain, problem, plan)
			@set ds.x = deduplicate(ds.x)
		end
		log_value(logger, "time_minibatch", t; step=0)
		opt = AdaBelief();
		ps = Flux.params(model);
		t = @elapsed train!(NeuroPlanner.loss, model, ps, opt, () -> rand(minibatches), max_steps; logger, trn_data = minibatches)
		log_value(logger, "time_train", t; step=0)
		serialize(filename*"_model.jls", model)	
		model
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
	serialize(filename*"_stats.jls", stats)
	settings !== nothing && serialize(filename*"_settings.jls",settings)
end

"""
ArgParse example implemented in Comonicon.

# Arguments

- `problem_name`: a name of the problem to solve ("ferry", "gripper", "blocks", "npuzzle")
- `arch_name`: an architecture of the neural network implementing heuristic("asnet", "pddl")
- `loss_name`: 

# Options

- `--max_steps <Int>`: maximum number of steps of SGD algorithm (default 10_000)
- `--max_time <Int>`:  maximum steps of the planner used for evaluation (default 30)
- `--graph_layers <Int>`:  maximum number of layers of (H)GNN (default 1)
- `--dense_dim <Int>`:  dimension of all hidden layers, even those realizing graph convolutions (default  32)
- `--dense_layers <Int>`:  number of layers of dense network after pooling vertices (default 32)
- `--residual <String>`:  residual connections between graph convolutions (none / dense / linear)

max_steps = 10_000; max_time = 30; graph_layers = 1; dense_dim = 32; dense_layers = 2; residual = "none"; seed = 1
domain_name = "elevators_00"
loss_name = "lstar"
arch_name = "hgnn"
"""

@main function main(domain_name, arch_name, loss_name; max_steps::Int = 10_000, max_time::Int = 30, graph_layers::Int = 1, 
		dense_dim::Int = 32, dense_layers::Int = 2, residual::String = "none", seed::Int = 1)
	Random.seed!(seed)
	settings = (;domain_name, arch_name, loss_name, max_steps, max_time, graph_layers, dense_dim, dense_layers, residual, seed)
	archs = Dict("asnet" => ASNet, "pddl" => HyperExtractor, "hgnnlite" => HGNNLite, "hgnn" => HGNN)
	residual = Symbol(residual)
	domain_pddl, problem_files = getproblem(domain_name, false)
	# problem_files = filter(s -> isfile(plan_file(domain_name, s)), problem_files)
	train_files = filter(s -> isfile(plan_file(s)), problem_files)
	train_files = sample(train_files, min(div(length(problem_files), 2), length(train_files)), replace = false)
	fminibatch = NeuroPlanner.minibatchconstructor(loss_name)
	hnet = archs[arch_name]

	filename = joinpath("super", domain_name, join([arch_name, loss_name, max_steps,  max_time, graph_layers, residual, dense_layers, dense_dim, seed], "_"))
	experiment(domain_name, hnet, domain_pddl, train_files, problem_files, filename, fminibatch; max_steps, max_time, graph_layers, residual, dense_layers, dense_dim, settings)
end