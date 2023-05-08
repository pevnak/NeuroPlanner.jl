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
include("utils.jl")

function experiment(domain_pddl, hnet, problem_files, fminibatch, planner; max_steps, max_time, graph_layers, dense_dim, dense_layers, residual, opt_type, epsilon, max_loss, sort_by_complexity, artificial_goals, double_maxtime, solve_solved, stop_after, max_epochs, max_sim_time, filename)
	isempty(filename) || isdir(dirname(filename)) || mkpath(dirname(filename))
	opt_type = Symbol(opt_type)
	domain = load_domain(domain_pddl)
	pddld = hnet(domain; message_passes = graph_layers, residual)
	solutions = Vector{Any}(fill(nothing, length(problem_files)))

	#create model from some problem instance
	model = let 
		problem = load_problem(first(problem_files))
		pddle, state = initproblem(pddld, problem)
		h₀ = pddle(state)
		model = reflectinmodel(h₀, d -> ffnn(d, dense_dim, dense_dim, dense_layers);fsm = Dict("" =>  d -> ffnn(d, dense_dim, 1, dense_layers)))
	end

	offset = 1
	# opt = Flux.Optimise.Optimiser(AdaBelief(), WeightDecay(1f-4));
	opt = AdaBelief()
	ps = Flux.params(model)
	all_solutions = []
	losses = []
	fvals = fill(typemax(Float64), length(solutions))
	start_time = time()
	logger=tblogger(filename*"_events.pb")
	for epoch in 1:max_epochs
		solved_before = issolved.(solutions)
		offset = 1
		while offset < length(solutions)
			log_value(logger, "offset", offset)
			offset, updated = update_solutions!(solutions, pddld, model, problem_files, dedup_fmb ∘ fminibatch, planner; offset, solve_solved, stop_after, max_time, artificial_goals)
			# print some Statistics
			solved = findall(issolved.(solutions))
			print("offset = ", offset," updated = ", length(updated), " ")
			log_value(logger, "solved", length(solved))
			show_stats(solutions)
			length(solved) == length(solutions) && break
			
			ii = [s !== nothing for s in solutions]
			t₁ = @elapsed fvals[ii] .= train!(NeuroPlanner.loss, model, ps, opt, solutions[ii], fvals[ii], max_steps; max_loss, ϵ = epsilon, opt_type = Symbol(opt_type))
			l = filter(x -> x != typemax(Float64), fvals)
			log_value(logger, "mean error", mean(l))
			log_value(logger, "worst case", maximum(l))
			println("epoch = $(epoch) offset $(offset)  mean error = ", mean(l), " worst case = ", maximum(l)," time: = ", t₁)
			time() - start_time > max_sim_time && break
		end
		solved_after = issolved.(solutions)
		log_value(logger, "solved after epoch", sum(solved_after))
		push!(losses, deepcopy(fvals))
		push!(all_solutions, map(s -> issolved(s) ? s.stats : nothing , solutions))
		if !isempty(filename)
			serialize(filename*"_stats.jls", (;losses, all_solutions, model))
		end
		all(issolved.(solutions)) && break
		if sum(solved_after) == sum(solved_before)  && double_maxtime
			max_time *= 2
		end

		time() - start_time > max_sim_time && break
	end
	all_solutions
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

max_steps = 10_000; max_time = 30; graph_layers = 1; dense_dim = 32; dense_layers = 2; residual = "none"; seed = 1;  opt_type = "mean"; epsilon = 0.0; max_loss = 0.0; sort_by_complexity = false; artificial_goals = false; double_maxtime = false; solve_solved = false; stop_after = 32; max_epochs = 100; max_sim_time = Int(23.5*3600)
domain_name = "ferry"
loss_name = "l2"
arch_name = "hgnn"
planner_name = "astar"
"""

@main function main(domain_name, arch_name, loss_name; max_steps::Int = 10_000, max_time::Int = 30, graph_layers::Int = 2, 
		dense_dim::Int = 16, dense_layers::Int = 2, residual::String = "none", seed::Int = 1, planner_name::String = "astar", 
		opt_type::String = "mean", epsilon::Float64 = 0.0, max_loss::Float64 = 0.0, sort_by_complexity::Bool = false, 
		artificial_goals::Bool = false, double_maxtime::Bool = false, solve_solved::Bool = false, stop_after::Int = 32,
		max_epochs::Int = 10, max_sim_time::Int = Int(23.5*3600))
	Random.seed!(seed)
	settings = (;domain_name, arch_name, loss_name, planner_name, max_steps, max_time, graph_layers, dense_dim, dense_layers, residual, opt_type, epsilon, max_loss, sort_by_complexity, artificial_goals, double_maxtime, solve_solved, stop_after, seed)
	archs = Dict("asnet" => ASNet, "pddl" => HyperExtractor, "hgnnlite" => HGNNLite, "hgnn" => HGNN, "levinasnet" => LevinASNet)
	planners = Dict("astar" => AStarPlanner, "bfs" => BFSPlanner,)
	planner = planners[planner_name]
	residual = Symbol(residual)
	domain_pddl, problem_files = getproblem(domain_name, false)
	fminibatch = NeuroPlanner.minibatchconstructor(loss_name)
	hnet = archs[arch_name]
	hid = hash((max_steps, max_time, graph_layers, dense_dim, dense_layers, residual, opt_type, epsilon, max_loss, sort_by_complexity, artificial_goals, double_maxtime, solve_solved, stop_after, max_epochs, max_sim_time))
	filename = joinpath("selflearning", domain_name, join([arch_name, loss_name, planner_name, hid, seed], "_"))
	isdir(dirname(filename)) || mkpath(dirname(filename))
	open(fio -> print(fio, JSON.json(settings)), filename*"_config.json","w")
	experiment(domain_pddl, hnet, problem_files, fminibatch, planner; max_steps, max_time, graph_layers, dense_dim, dense_layers, residual, opt_type, epsilon, max_loss, sort_by_complexity, artificial_goals, double_maxtime, solve_solved, stop_after, max_epochs, max_sim_time, filename)
end