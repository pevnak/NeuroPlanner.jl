using NeuroPlanner
using PDDL
using Flux
using JSON
using GraphNeuralNetworks
using SymbolicPlanners
using PDDL: GenericProblem
using SymbolicPlanners: PathSearchSolution
using Statistics
using IterTools
using Random
using StatsBase
using Serialization

include("solution_tracking.jl")
include("problems.jl")
include("losses.jl")
include("training.jl")

function ffnn(d, odim, nlayers)
	nlayers == 1 && return(Dense(d,1))
	nlayers == 2 && return(Chain(Dense(d, odim, relu), Dense(odim,1)))
	nlayers == 3 && return(Chain(Dense(d, odim, relu), Dense(odim, odim, relu), Dense(odim,1)))
	error("nlayers should be only in [1,3]")
end

######
# define a NN based solver
######
struct GNNHeuristic{P,M} <: Heuristic 
	pddle::P
	model::M
end

GNNHeuristic(pddld, problem, model) = GNNHeuristic(NeuroPlanner.add_goalstate(pddld, problem), model)
Base.hash(g::GNNHeuristic, h::UInt) = hash(g.model, hash(g.pddle, h))
SymbolicPlanners.compute(h::GNNHeuristic, domain::Domain, state::State, spec::Specification) = only(h.model(h.pddle(state)))

function experiment(domain_pddl, problem_files, ofile, loss_fun, fminibatch, planner; opt_type = :worst, solve_solved = false, stop_after=32, filename = "", max_time=30, double_maxtime=false, max_steps = 100, max_loss = 0.0, epsilon = 0.5, artificial_goals = false, graph_layers = 2, graph_dim = 8, dense_layers = 2, dense_dim = 32, max_epochs = 100, max_sim_time = 23.5*3600, configuration = nothing)
	isdir(ofile()) || mkpath(ofile())
	domain = load_domain(domain_pddl)
	pddld = PDDLExtractor(domain)
	solutions = Vector{Any}(fill(nothing, length(problem_files)))

	#create model from some problem instance
	model = let 
		problem = load_problem(first(problem_files))
		pddle, state = initproblem(pddld, problem)
		h₀ = pddle(state)
		model = MultiModel(h₀, graph_dim, graph_layers, d -> ffnn(d, dense_dim, dense_layers))
		solve_problem(pddld, problem, model, planner)	#warmup the solver
		model
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
			offset, updated = update_solutions!(solutions, pddld, model, problem_files, fminibatch, planner; offset, solve_solved, stop_after, max_time, artificial_goals)
			# print some Statistics
			solved = findall(issolved.(solutions))
			print("offset = ", offset," updated = ", length(updated), " ")
			show_stats(solutions)
			length(solved) == length(solutions) && break
			
			ii = [s !== nothing for s in solutions]
			t₁ = @elapsed fvals[ii] .= train!(x -> loss_fun(model, x), model, ps, opt, solutions[ii], fvals[ii], max_steps; max_loss, ϵ = epsilon, opt_type = Symbol(opt_type))
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

# Let's make configuration ephemeral
problem_name = ARGS[1]
configuration = ARGS[2]

# problem_name = "ferry"
# configuration = 1

# problem_name = "ferry"
# seed = 1
# double_maxtime = false
# loss_name = "lgbfs"
# planner_name = "gbfs"
# loss_name = "lstar"
# planner_name = "astar"
# solve_solved = false
# stop_after = 32
# max_steps = 2000
# opt_type = :worst
# opt_type = :mean
# epsilon = 0.5
# max_loss = 0.0
# max_time = 30
# graph_layers = 2
# graph_dim = 8
# dense_layers = 2
# dense_dim = 32
# sort_by_complexity = true
# artificial_goals = false

js = open(JSON.parse, "configurations/$(configuration).json", "r")
seed = js["seed"]
double_maxtime = js["double_maxtime"]
loss_name = js["loss_name"]
planner_name = js["planner_name"]
solve_solved = js["solve_solved"]
stop_after = js["stop_after"]
max_steps = js["max_steps"]
opt_type = js["opt_type"]
epsilon = js["epsilon"]
max_loss = js["max_loss"]
max_time = js["max_time"]
graph_layers = js["graph_layers"]
graph_dim = js["graph_dim"]
dense_layers = js["dense_layers"]
dense_dim = js["dense_dim"]
sort_by_complexity = js["sort_by_complexity"]
artificial_goals = js["artificial_goals"]

Random.seed!(seed)
domain_pddl, problem_files, ofile = getproblem(problem_name, sort_by_complexity)
planner = (planner_name == "astar") ? AStarPlanner : GreedyPlanner
loss_fun, fminibatch = getloss(loss_name)
filename = ofile("$(configuration).jls")
experiment(domain_pddl, problem_files, ofile, loss_fun, fminibatch, planner; double_maxtime, filename, solve_solved, stop_after, max_steps, max_loss, max_time, opt_type, epsilon, artificial_goals, configuration = js)


function generate_configurations()
	configurations = Set()
	while length(configurations) < 100
		js = Dict("seed" => 1,
	       "double_maxtime" => rand([true,false]),
	       "loss_name" => "lstar",
	       "planner_name" => "astar",
	       "solve_solved" => rand([true,false]),
	       "stop_after" => rand(2 .^ (0:5)),
	       "max_steps" => rand([100, 500, 1000, 2000, 4000]),
	       "opt_type" => rand([:mean, :worst]),
	       "epsilon" => rand([0.0, 0.1, 0.3 , 0.5]),
	       "max_loss" => rand([0.0, 0.1, 0.2, 0.3]),
	       "max_time" => rand([30, 60, 120]),
	       "graph_layers" => rand([1,2,3]),
	       "graph_dim" => rand([4, 8, 16, 32]),
	       "dense_layers" => rand([1,2,3]),
	       "dense_dim" => rand([4, 8, 16, 32, 64]),
	       "sort_by_complexity" => rand([true,false]),
	       "artificial_goals" => rand([true,false]),
	       )
		js ∈ configurations && continue
		push!(configurations, js)
		open("configurations/$(length(configurations)).json","w") do fio
			JSON.print(fio, js)
		end
		@show length(configurations)
	end
end