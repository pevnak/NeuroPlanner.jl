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
using NeuroPlanner: sample_backward_trace, search_tree_from_trajectory

include("solution_tracking.jl")
include("problems.jl")
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

########
#	define function for generating minibatch from reversed plans
########
function prepare_minibatch(pddld, fminibatch, bs::BackwardSampler)
	pddld.domain != bs.domain && error("domain of pddl has to be the same as of the sampler")
	domain = bs.domain
	problem = bs.problem
	trajectory, plan = bs()
	st = search_tree_from_trajectory(domain, trajectory, plan)
	pddle = NeuroPlanner.add_goalstate(pddld, problem, trajectory[end])
	fminibatch(st, pddle, trajectory)
end


function prepare_minibatch(pddld, fminibatch, problem_file::AbstractString)
	prepare_minibatch(pddld, fminibatch, load_problem(problem_file))
end

function prepare_minibatch(pddld, fminibatch, problem)
	domain = pddld.domain
	trajectory, plan = sample_backward_trace(domain, problem, rand(1:30))
	st = search_tree_from_trajectory(domain, trajectory, plan)
	pddle = NeuroPlanner.add_goalstate(pddld, problem, trajectory[end])
	fminibatch(st, pddle, trajectory)
end

function test_planner(pddld, model, problem_files, planner)
	isempty(problem_files) && return(nothing)
	solutions = Vector{Any}(fill(nothing, length(problem_files)))
	update_solutions!(solutions, pddld, model, problem_files, (args...) -> nothing, planner; offset = 1, stop_after = length(problem_files), max_time)
	[issolved(s) ? s.stats : s for s in solutions]
end

function experiment(domain_pddl, train_files, test_files, ofile, loss_fun, fminibatch; 
			filename = "", max_time=30, max_steps = 10000, sample_goal = true,
			graph_layers = 2, graph_dim = 8, dense_layers = 2, 
			dense_dim = 32, max_epochs = 100, max_states = 100_000)
	isdir(ofile()) || mkpath(ofile())
	domain = load_domain(domain_pddl)
	pddld = PDDLExtractor(domain)

	#create model from some problem instance
	model = let 
		problem = load_problem(first(train_files))
		pddle, state = initproblem(pddld, problem)
		h₀ = pddle(state)
		model = MultiModel(h₀, graph_dim, graph_layers, d -> ffnn(d, dense_dim, dense_layers))
		planner = (heuristic; kwargs...) -> ForwardPlanner(;heuristic=heuristic, h_mult=1, kwargs...)
		solve_problem(pddld, problem, model, planner)	#warmup the solver
		model
	end

	# problems = map(load_problem, train_files)
	samplers = mapreduce(vcat, train_files) do f 
		BackwardSampler(domain, load_problem(f); max_states, sample_goal)
	end
	fvals = map(1:100) do i
		ms = div(max_steps, 100)
		n = ceil(Int, ms / length(train_files))
		# training_set = mapreduce(vcat, train_files) do f 
		# 	bs = BackwardSampler(domain, load_problem(f); max_states, sample_goal)
		# 	[prepare_minibatch(pddld, fminibatch, bs) for _ in 1:n]
		# end
		# pm() = rand(training_set)
	
		pm() = prepare_minibatch(pddld, fminibatch, bs)
		# pm() = prepare_minibatch(pddld, fminibatch, rand(problems))
		opt = AdaBelief()
		ps = Flux.params(model)
		fval = train!(x -> loss_fun(model, x), model, ps, opt, pm, ms)
		println(i, ": ", fval)
		fval
	end

	serialize(filename[1:end-4]*"_model.jls", model)

	results = []
	for h_mult in [1, 1.5, 2, 2.5]
		planner = (heuristic; kwargs...) -> ForwardPlanner(;heuristic=heuristic, h_mult=h_mult, kwargs...)
		train_solutions = test_planner(pddld, model, train_files, planner)
		test_solutions = test_planner(pddld, model, test_files, planner)
		r = (;h_mult, train_solutions, test_solutions)
		push!(results, r)
		serialize(filename, results)
	end
end

# Let's make configuration ephemeral
# problem_name = "gripper"; loss_name = "lstar";seed = 1

problem_name = ARGS[1]
loss_name = ARGS[2]
seed = parse(Int,ARGS[3])

max_time = 30
graph_layers = 2
graph_dim = 8
dense_layers = 2
dense_dim = 32
max_steps = 100_000
max_states = 1_000_000
# sample_goal = true
sample_goal = false
ttratio = 1.0

# max_steps = 1_000
# max_states = 10_000

Random.seed!(seed)
domain_pddl, problem_files, _ = getproblem(problem_name, false)
train_files = sample(problem_files, round(Int, ttratio*length(problem_files)); replace = false)
test_files = setdiff(problem_files, train_files)
loss_fun, fminibatch = getloss(loss_name)
ofile(s...) = joinpath("deepcubea3", problem_name, s...)
filename = ofile(join([loss_name, max_time, graph_layers, graph_dim, dense_layers, dense_dim, max_steps,  max_states, ttratio, seed, sample_goal], "_")*".jls")
experiment(domain_pddl, train_files, test_files, ofile, loss_fun, fminibatch; filename, max_time, graph_layers, graph_dim, dense_layers, dense_dim, max_steps, max_states, sample_goal)

