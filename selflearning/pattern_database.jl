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

"""
	This is pretty lame and should be done better (one day if it works)
"""
function create_training_set(pddld, problem_file, fminibatch, n, depths; forward_traces = true, backward_traces = true)
	problem = load_problem(problem_file)
	domain = pddld.domain
	map(1:n) do _
		depth = rand(depths)
		trajectory, plan = sample_trace(domain, problem, depth, forward_traces, backward_traces)
		search_tree = search_tree_from_trajectory(domain, trajectory, plan)
		pddle = NeuroPlanner.add_goalstate(pddld, problem, trajectory[end])
		fminibatch(search_tree, pddle, trajectory)
	end
end

function test_planner(pddld, model, problem_file, planner; max_time = 30)
	solutions = Vector{Any}(fill(nothing, 1))
	update_solutions!(solutions, pddld, model, [problem_file], (args...) -> nothing, planner; offset = 1, stop_after = length(problem_files), max_time)
	only(solutions)
end

function experiment(domain_pddl, problem_file, ofile, loss_fun, fminibatch; 
			opt_type = :worst, filename = "", max_time=30, max_steps = 10000, 
			max_loss = 0.0, epsilon = 0.5, graph_layers = 2, graph_dim = 8, dense_layers = 2, 
			dense_dim = 32, max_epochs = 100, training_set_size = 10000, depths = 1:30)
	isdir(ofile()) || mkpath(ofile())
	domain = load_domain(domain_pddl)
	pddld = PDDLExtractor(domain)

	#create model from some problem instance
	model = let 
		problem = load_problem(problem_file)
		pddle, state = initproblem(pddld, problem)
		h₀ = pddle(state)
		model = MultiModel(h₀, graph_dim, graph_layers, d -> ffnn(d, dense_dim, dense_layers))
		planner = (heuristic; kwargs...) -> ForwardPlanner(;heuristic=heuristic, h_mult=1, kwargs...)
		solve_problem(pddld, problem, model, planner)	#warmup the solver
		model
	end

	training_set = create_training_set(pddld, problem_file, fminibatch, training_set_size, depths)
	opt = AdaBelief()
	ps = Flux.params(model)
	fvals = fill(typemax(Float64), length(training_set))
	fvals .= train!(x -> loss_fun(model, x), model, ps, opt, training_set, fvals, max_steps; max_loss, ϵ = epsilon, opt_type = Symbol(opt_type))

	# serialize(filename[end-4:end]*"_model.jls", model) 

	results = []
	for h_mult in [1, 1.5, 2, 2.5]
		planner = (heuristic; kwargs...) -> ForwardPlanner(;heuristic=heuristic, h_mult=h_mult, kwargs...)
		solution = only(test_planner(pddld, model, [problem_file], planner; max_time))
		r = (;h_mult, solution)
		push!(results, r)
		# serialize(filename, results)
	end
end

# Let's make configuration ephemeral
# problem_name = ARGS[1]
# problem_name = "gripper"
# loss_name = "lstar"
# problem_file = "benchmarks/gripper/problems/gripper-n45.pddl"
# seed = 1

problem_name = ARGS[1]
loss_name = ARGS[2]
seed = parse(Int,ARGS[3])

opt_type = :mean
epsilon = 0.5
max_time = 30
graph_layers = 2
graph_dim = 8
dense_layers = 2
dense_dim = 32
training_set_size = 100000
max_steps = 10*training_set_size
max_loss = 0.0
depths = 1:30
ttratio = 1.0

Random.seed!(seed)
domain_pddl, problem_files, _ = getproblem(problem_name, false)
train_files = sample(problem_files, round(Int, length(problem_files)); replace = false)
test_files = setdiff(problem_files, train_files)
loss_fun, fminibatch = getloss(loss_name)
ofile(s...) = joinpath("deepcubea3", problem_name, s...)
filename = ofile(join([loss_name,opt_type,epsilon,max_time,graph_layers,graph_dim,dense_layers,dense_dim,training_set_size,max_steps,max_loss, depths,ttratio,seed], "_")*".jls")
experiment(domain_pddl, train_files, test_files, ofile, loss_fun, fminibatch; filename, opt_type, epsilon, max_time, graph_layers, graph_dim, dense_layers, dense_dim, training_set_size, max_steps, depths, max_loss)
