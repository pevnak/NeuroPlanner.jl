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
using Logging
using TensorBoardLogger

include("solution_tracking.jl")
include("problems.jl")
include("training.jl")
include("utils.jl")

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
			@show problem_file
			println("creating sample from problem: ",problem_file)
			plan = load_plan(problem_file)
			problem = load_problem(problem_file)
			ds = fminibatch(pddld, domain, problem, plan)
			dedu = @set ds.x = deduplicate(ds.x)
			size_o, size_d =  Base.summarysize(ds), Base.summarysize(dedu)
			println("original: ", size_o, " dedupped: ", size_d, " (",round(100*size_d / size_o, digits =2),"%)")
			dedu
		end
		log_value(logger, "time_minibatch", t; step=0)
		opt = AdaBelief();
		ps = Flux.params(model);
		t = @elapsed train!(NeuroPlanner.loss, model, ps, opt, () -> rand(minibatches), max_steps; logger, trn_data = minibatches)
		log_value(logger, "time_train", t; step=0)
		serialize(filename*"_model.jls", model)	
		model
	end
	planners = model isa NeuroPlanner.LevinModel ? [BFSPlanner] : [AStarPlanner, GreedyPlanner, BackwardPlannerAStarPlanner, BackwardGreedyPlanner]

	stats = map(Iterators.product(planners, problem_files)) do (planner, problem_file)
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

max_steps = 10_000; max_time = 30; graph_layers = 2; dense_dim = 16; dense_layers = 2; residual = "none"; seed = 1
domain_name = "ferry"
loss_name = "lstar"
arch_name = "pddl"
"""

@main function main(domain_name, arch_name, loss_name; max_steps::Int = 10_000, max_time::Int = 30, graph_layers::Int = 1, 
		dense_dim::Int = 32, dense_layers::Int = 2, residual::String = "none", seed::Int = 1)
	Random.seed!(seed)
	settings = (;domain_name, arch_name, loss_name, max_steps, max_time, graph_layers, dense_dim, dense_layers, residual, seed)
	@show settings
	archs = Dict("asnet" => ASNet, "pddl" => HyperExtractor, "hgnnlite" => HGNNLite, "hgnn" => HGNN, "levinasnet" => LevinASNet)
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


function check_optimality()
	plan = load_plan(problem_file)
	problem = load_problem(problem_file)

	s = goalstate(domain, problem)
	for sol_a in reverse(plan)
		hs = map(relevant(domain, s)) do a
			s₋₁ = PDDL.regress(domain, s, a)
			h = only(hfun.model(hfun.pddle(s₋₁)))
			(h, a == sol_a)
		end

		if argmin(first, hs)[2]
			for (h, a) in hs
				a ? print(@yellow "$(h) ") : print(h, " ")
			end
		else
			for (h, a) in hs
				a ? print(@red "$(h) ") : print(h, " ")
			end
		end
		println()
		s = PDDL.regress(domain, s, sol_a)
	end

	only(hfun.model(hfun.pddle(goalstate(domain, problem))))


	# Let's run the planner
	hfun = EvalTracker(NeuroHeuristic(NeuroPlanner.add_initstate(pddld, problem), model, Ref(0.0)))
	planner = BackwardAStarPlanner(hfun; max_time = 30, save_search = true)
	sol = planner(domain, problem)
	plot_search_tree("/tmp/backward", sol, hfun)
end

