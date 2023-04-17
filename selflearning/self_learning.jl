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

function dedup_fmb(dedup_model, fminibatch, args...)
	ds = fminibatch(args...)
	@set ds.x = deduplicate(dedup_model, ds.x)
end

function experiment(domain_pddl, problem_files, fminibatch, planner; opt_type = :worst, solve_solved = false, stop_after=32, filename = "", max_time=30, residual = :linear, double_maxtime=false, max_steps = 100, max_loss = 0.0, epsilon = 0.5, artificial_goals = false, graph_layers = 2,  dense_layers = 2, dense_dim = 32, max_epochs = 100, max_sim_time = 23.5*3600, configuration = nothing)
	isempty(filename) || isdir(dirname(filename)) || mkpath(dirname(filename))
	domain = load_domain(domain_pddl)
	pddld = HyperExtractor(domain; message_passes = graph_layers, residual)
	solutions = Vector{Any}(fill(nothing, length(problem_files)))

	#create model from some problem instance
	model, dedup_model = let 
		problem = load_problem(first(problem_files))
		pddle, state = initproblem(pddld, problem)
		h₀ = pddle(state)
		model = reflectinmodel(h₀, d -> ffnn(d, dense_dim, dense_dim, dense_layers);fsm = Dict("" =>  d -> ffnn(d, dense_dim, 1, dense_layers)))
		# model = reflectinmodel(h₀, d -> Dense(d, dense_dim, relu);fsm = Dict("" =>  d -> ffnn(d, dense_dim, 1, dense_layers)))
		dedup_model = reflectinmodel(h₀, d -> Dense(d,32) ;fsm = Dict("" =>  d -> Dense(d, 32)))
		model, dedup_model
	end

	offset = 1
	# opt = Flux.Optimise.Optimiser(AdaBelief(), WeightDecay(1f-4));
	opt = AdaBelief()
	ps = Flux.params(model)
	all_solutions = []
	losses = []
	fvals = fill(typemax(Float64), length(solutions))
	times = [time()]
	start_time = time()
	for epoch in 1:max_epochs
		solved_before = issolved.(solutions)
		offset = 1
		while offset < length(solutions)
			offset, updated = update_solutions!(solutions, pddld, model, problem_files, (args...) -> dedup_fmb(dedup_model, fminibatch, args...), planner; offset, solve_solved, stop_after, max_time, artificial_goals)
			# print some Statistics
			solved = findall(issolved.(solutions))
			print("offset = ", offset," updated = ", length(updated), " ")
			show_stats(solutions)
			length(solved) == length(solutions) && break
			
			ii = [s !== nothing for s in solutions]
			t₁ = @elapsed fvals[ii] .= train!(NeuroPlanner.loss, model, ps, opt, solutions[ii], fvals[ii], max_steps; max_loss, ϵ = epsilon, opt_type = Symbol(opt_type))
			l = filter(x -> x != typemax(Float64), fvals)
			println("epoch = $(epoch) offset $(offset)  mean error = ", mean(l), " worst case = ", maximum(l)," time: = ", t₁)
			time() - start_time > max_sim_time && break
		end
		push!(times, time())
		solved_after = issolved.(solutions)
		push!(losses, deepcopy(fvals))
		push!(all_solutions, map(s -> issolved(s) ? s.stats : nothing , solutions))
		if !isempty(filename)
			serialize(filename,(;all_solutions, losses))
		end
		all(issolved.(solutions)) && break
		if sum(solved_after) == sum(solved_before)  && double_maxtime
			max_time *= 2
		end
		serialize(filename, (;times, losses, all_solutions, model, configuration))
		time() - start_time > max_sim_time && break
	end
	all_solutions
end

# js = open(JSON.parse, "configurations/$(configuration).json", "r")
# seed = js["seed"]
# double_maxtime = js["double_maxtime"]
# loss_name = js["loss_name"]
# planner_name = js["planner_name"]
# solve_solved = js["solve_solved"]
# stop_after = js["stop_after"]
# max_steps = js["max_steps"]
# opt_type = js["opt_type"]
# epsilon = js["epsilon"]
# max_loss = js["max_loss"]
# max_time = js["max_time"]
# graph_layers = js["graph_layers"]
# graph_dim = js["graph_dim"]
# dense_layers = js["dense_layers"]
# dense_dim = js["dense_dim"]
# sort_by_complexity = js["sort_by_complexity"]
# artificial_goals = js["artificial_goals"]


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
domain_name = "ferry"
loss_name = "l2"
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
	train_files = filter(s -> isfile(plan_file(domain_name, s)), problem_files)
	train_files = sample(train_files, min(div(length(problem_files), 2), length(train_files)), replace = false)
	fminibatch = NeuroPlanner.minibatchconstructor(loss_name)
	hnet = archs[arch_name]

	filename = joinpath("super", domain_name, join([arch_name, loss_name, max_steps,  max_time, graph_layers, residual, dense_layers, dense_dim, seed], "_"))
	experiment(domain_name, hnet, domain_pddl, train_files, problem_files, filename, fminibatch; max_steps, max_time, graph_layers, residual, dense_layers, dense_dim, settings)
end

Random.seed!(seed)
domain_pddl, problem_files = getproblem(domain_name, sort_by_complexity)
planner = (planner_name == "astar") ? AStarPlanner : GreedyPlanner
fminibatch = NeuroPlanner.minibatchconstructor(loss_name)
# filename = ofile("$(configuration).jls")
filename = joinpath("selflearning", domain_name, join([loss_name, max_steps,  max_time, graph_layers, residual, dense_layers, dense_dim, seed], "_")*".jls")
experiment(domain_pddl, problem_files, fminibatch, planner; double_maxtime, filename, solve_solved, stop_after, max_steps, max_loss, max_time, opt_type, residual, epsilon, artificial_goals, graph_layers, dense_layers, dense_dim)


# function generate_configurations()
# 	configurations = Set()
# 	while length(configurations) < 100
# 		js = Dict("seed" => 1,
# 	       "double_maxtime" => rand([true,false]),
# 	       "loss_name" => "lstar",
# 	       "planner_name" => "astar",
# 	       "solve_solved" => rand([true,false]),
# 	       "stop_after" => rand(2 .^ (0:5)),
# 	       "max_steps" => rand([100, 500, 1000, 2000, 4000]),
# 	       "opt_type" => rand([:mean, :worst]),
# 	       "epsilon" => rand([0.0, 0.1, 0.3 , 0.5]),
# 	       "max_loss" => rand([0.0, 0.1, 0.2, 0.3]),
# 	       "max_time" => rand([30, 60, 120]),
# 	       "graph_layers" => rand([1,2,3]),
# 	       "graph_dim" => rand([4, 8, 16, 32]),
# 	       "dense_layers" => rand([1,2,3]),
# 	       "dense_dim" => rand([4, 8, 16, 32, 64]),
# 	       "sort_by_complexity" => rand([true,false]),
# 	       "artificial_goals" => rand([true,false]),
# 	       )
# 		js ∈ configurations && continue
# 		push!(configurations, js)
# 		open("configurations/$(length(configurations)).json","w") do fio
# 			JSON.print(fio, js)
# 		end
# 		@show length(configurations)
# 	end
# end