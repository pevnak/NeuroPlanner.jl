using NeuroPlanner
using PDDL
using Flux
using JSON
using CSV
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
using Logging
using TensorBoardLogger
using Comonicon

include("solution_tracking.jl")
include("problems.jl")
include("training.jl")
include("utils.jl")

function experiment(domain_name, hnet, domain_pddl, train_files, problem_files, filename, fminibatch;max_steps = 10000, max_time = 30, graph_layers = 2, residual = true, dense_layers = 2, dense_dim = 32, settings = nothing, aggregation = SegmentedSumMax)
	!isdir(dirname(filename)) && mkpath(dirname(filename))
	domain = load_domain(domain_pddl)
	pddld = hnet(domain; message_passes = graph_layers, residual)
	#create model from some problem instance

	# we can check that the model can learn the training data if the dedup_model can differentiate all input states, which is interesting by no means
	# A special hook to rerun the faster mixed lrnn2
	modelfile = filename*"_model.jls"
	model = if isfile(modelfile)
		deserialize(modelfile)
	else
		model = let 
			problem = load_problem(first(train_files))
			pddle, state = initproblem(pddld, problem)
			h₀ = pddle(state)
			reflectinmodel(h₀, d -> Dense(d, dense_dim, relu), aggregation;fsm = Dict("" =>  d -> ffnn(d, dense_dim, 1, dense_layers)))
		end

		t = @elapsed minibatches = map(train_files) do problem_file
			try
				println("creating sample from problem: ",problem_file)
				plan = load_plan(problem_file)
				problem = load_problem(problem_file)
				ds = fminibatch(pddld, domain, problem, plan)
				dedu = @set ds.x = deduplicate(ds.x)
				size_o, size_d =  Base.summarysize(ds), Base.summarysize(dedu)
				println("original: ", size_o, " dedupped: ", size_d, " (",round(100*size_d / size_o, digits =2),"%)")
				dedu
			catch 
				println("there was a problem with ", problem_file)
				return missing
			end
		end
		minibatches = collect(skipmissing(minibatches))
		logger=TBLogger(filename*"_events")
		log_value(logger, "time_minibatch", t; step=0)
		opt = AdaBelief();
		ps = Flux.params(model);
		t = @elapsed train!(NeuroPlanner.loss, model, ps, opt, () -> rand(minibatches), max_steps; logger, trn_data = minibatches, reset_fval=1000)
		log_value(logger, "time_train", t; step=0)
		serialize(modelfile, model)	
		model
	end
	planners = model isa NeuroPlanner.LevinModel ? [BFSPlanner] : [AStarPlanner, GreedyPlanner]

	stats = DataFrame()
	#precompilation
	# solve_problem(pddld, first(problem_files), model, first(planners); return_unsolved = true, max_time)
	t₀ = time()
	for (planner, problem_file) in Iterators.product(planners, problem_files)
		try 
			used_in_train = problem_file ∈ train_files
			@show problem_file
			t = @elapsed sol = solve_problem(pddld, problem_file, model, planner; return_unsolved = true, max_time)
			println("time in the solver: ", t, " status:  ", sol.sol.status, " length: ", length(sol.sol.trajectory) - 1)
			trajectory = sol.sol.status == :max_time ? nothing : sol.sol.trajectory
			s = merge(sol.stats, (;used_in_train, planner = "$(planner)", trajectory, problem_file))
			# @show s
			push!(stats, s, cols=:union, promote=true)
			if time()-t₀ > 3600	# serialize stats evert hour
				serialize(filename*"_stats_tmp.jls", stats)
				t₀ = time()
			end
		catch
			println("failed on ",problem_file)
		end
	end
	println("evaluation finished")
	serialize(filename*"_stats.jls", stats)
	rm(filename*"_stats_tmp.jls")
	settings !== nothing && serialize(filename*"_settings.jls",settings)
end

"""
ArgParse example implemented in Comonicon.

# Arguments
- `problem_name`: a name of the problem to solve ("ferry", "gripper", "blocks", "npuzzle", "elevator_00")
- `arch_name`: an architecture of the neural network implementing heuristic("asnet", "pddl")
- `loss_name`: 

# Options
- `--max_steps <Int>`: maximum number of steps of SGD algorithm (default 10_000)
- `--max_time <Int>`:  maximum steps of the planner used for evaluation (default 30)
- `--graph_layers <Int>`:  maximum number of layers of (H)GNN (default 1)
- `--dense_dim <Int>`:  dimension of all hidden layers, even those realizing graph convolutions (default  32)
- `--dense_layers <Int>`:  number of layers of dense network after pooling vertices (default 32)
- `--residual <String>`:  residual connections between graph convolutions (none / dense / linear)
- `--aggregation <String>`:  type of aggregation of the neighborhood ("meanmax / "summax")

# max_steps = 10_000; max_time = 30; aggregation = "summax"; graph_layers = 3; dense_dim = 16; dense_layers = 3; residual = "none"; seed = 1
# max_steps = 10_000; max_time = 30; aggregation = "summax"; graph_layers = 2; dense_dim = 16; dense_layers = 2; residual = "none"; seed = 1
# domain_name = "ferry"
# loss_name = "lstar"
# arch_name = "objectbinaryfe"
"""
@main function main(domain_name, arch_name, loss_name; max_steps::Int = 10_000, max_time::Int = 30, graph_layers::Int = 1, 
		dense_dim::Int = 32, aggregation = "summax", dense_layers::Int = 2, residual::String = "none", seed::Int = 1)
	Random.seed!(seed)
	settings = (;domain_name, arch_name, loss_name, max_steps, max_time, graph_layers, aggregation, dense_dim, dense_layers, residual, seed)
	@show settings
	filename = joinpath("super_amd_gnn", domain_name, join([arch_name, loss_name, max_steps,  max_time, graph_layers, aggregation, residual, dense_layers, dense_dim, seed], "_"))
	@show filename

	archs = Dict(
		"asnet" => ASNet,
		"hgnnlite" => HGNNLite,
		"hgnn" => HGNN,
		"levinasnet" => LevinASNet,
		"atombinaryfena"    => AtomBinaryFENA,
		"objectbinaryme"    => ObjectBinaryME,
		"objectatom"        => ObjectAtom,
		"objectatombipfe"   => ObjectAtomBipFE,
		"objectbinaryfena"  => ObjectBinaryFENA,
		"atombinaryme"      => AtomBinaryME,
		"objectatombipfena" => ObjectAtomBipFENA,
		"atombinaryfe"      => AtomBinaryFE,
		"objectbinaryfe"    => ObjectBinaryFE,
		"objectatombipme"   => ObjectAtomBipME,
		)
	aggregations = Dict("meanmax" => SegmentedMeanMax, "summax" => SegmentedSumMax)
	residual = Symbol(residual)
	domain_pddl, problem_files = getproblem(domain_name, false)
	# problem_files = filter(s -> isfile(plan_file(domain_name, s)), problem_files)
	train_files = filter(s -> isfile(plan_file(s)), problem_files)
	# train_files = domain_name ∉ IPC_PROBLEMS ? sample(train_files, min(div(length(problem_files), 2), length(train_files)), replace = false) : train_files
	fminibatch = NeuroPlanner.minibatchconstructor(loss_name)
	hnet = archs[arch_name]
	aggregation = aggregations[aggregation]
	experiment(domain_name, hnet, domain_pddl, train_files, problem_files, filename, fminibatch; max_steps, max_time, graph_layers, aggregation, residual, dense_layers, dense_dim,  settings)
end

