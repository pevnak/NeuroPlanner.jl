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
using DataFrames
using Mill
using Functors
using Setfield

include("solution_tracking.jl")
include("problems.jl")
include("training.jl")

function ffnn(idim, hdim, odim, nlayers)
	nlayers == 1 && return(Dense(idim,odim))
	nlayers == 2 && return(Chain(Dense(idim, hdim, relu), Dense(hdim,odim)))
	nlayers == 3 && return(Chain(Dense(idim, hdim, relu), Dense(hdim, hdim, relu), Dense(odim,odim)))
	error("nlayers should be only in [1,3]")
end

function plan_file(problem_file)
	middle_path = splitpath(problem_file)[3:end-1]
	middle_path = filter(∉(["problems"]),middle_path)
	middle_path = filter(∉(["test"]),middle_path)
	joinpath("plans", problem_name, middle_path..., basename(problem_file)[1:end-5]*".jls")
end

function experiment(domain_pddl, train_files, problem_files, filename, fminibatch;max_steps = 10000, max_time = 30, graph_layers = 2, graph_dim = 8, dense_layers = 2, dense_dim = 32)
	!isdir(dirname(filename)) && mkpath(dirname(filename))
	domain = load_domain(domain_pddl)
	# pddld = PDDLExtractor(domain)
	pddld = HyperExtractor(domain; message_passes = graph_layers)

	#create model from some problem instance
	model, dedup_model = let 
		problem = load_problem(first(problem_files))
		pddle, state = initproblem(pddld, problem)
		h₀ = pddle(state)
		model = reflectinmodel(h₀, d -> ffnn(d, dense_dim, dense_dim, dense_layers);fsm = Dict("" =>  d -> ffnn(d, dense_dim, 1, dense_layers)))
		dedup_model = reflectinmodel(h₀, d -> Dense(d,32) ;fsm = Dict("" =>  d -> Dense(d, 32)))
		model, dedup_model
	end

	minibatches = map(train_files) do problem_file
		plan = deserialize(plan_file(problem_file))
		problem = load_problem(problem_file)
		ds = fminibatch(pddld, domain, problem, plan.plan)
		# @set ds.x = deduplicate(dedup_model, ds.x)
		ds
	end

	if isfile(filename[1:end-4]*"_model.jls")
		model = deserialize(filename[1:end-4]*"_model.jls")
	else
		opt = AdaBelief()
		ps = Flux.params(model)
		train!(Base.Fix1(NeuroPlanner.loss, model), model, ps, opt, () -> rand(minibatches), max_steps)
		serialize(filename[1:end-4]*"_model.jls", model)	
	end


	# stats = map(Iterators.product([AStarPlanner, GreedyPlanner], problem_files)) do (planner, problem_file)
	stats = map(Iterators.product([AStarPlanner], problem_files)) do (planner, problem_file)
		used_in_train = problem_file ∈ train_files
		@show problem_file
		sol = solve_problem(pddld, problem_file, model, planner;return_unsolved = true)
		trajectory = sol.sol.status == :max_time ? nothing : sol.sol.trajectory
		merge(sol.stats, (;used_in_train, planner = "$(planner)", trajectory, problem_file))
	end
	df = DataFrame(vec(stats))
	mean(df.solved[.!df.used_in_train])
	serialize(filename, stats)
end

# Let's make configuration ephemeral
problem_name = ARGS[1]
loss_name = ARGS[2]
seed = parse(Int, ARGS[3])

problem_name = "ferry"
loss_name = "lstar"
seed = 1

max_steps = 20000
max_time = 30
graph_layers = 2
dense_layers = 2
dense_dim = 32
heads = 4
head_dims = 4

Random.seed!(seed)
domain_pddl, problem_files, ofile = getproblem(problem_name, false)
problem_files = filter(isfile ∘ plan_file, problem_files)
train_files = filter(isfile ∘ plan_file, problem_files)
train_files = sample(train_files, div(length(problem_files), 2), replace = false)
fminibatch = NeuroPlanner.minibatchconstructor(loss_name)

filename = joinpath("supervised_hyper2", problem_name, join([loss_name, max_steps,  max_time, graph_layers, dense_layers, dense_dim, seed], "_")*".jls")
experiment(domain_pddl, train_files, problem_files, filename, fminibatch; max_steps, max_time, graph_layers, dense_layers, dense_dim)
