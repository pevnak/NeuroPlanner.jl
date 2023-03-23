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


"""
struct HyperMHA{MI,MH}
	mill::MI
	mha::MH 
end

Combines hyper-network implemented as a one pass through the Mill model 
and one multi-head attention layer(though this one might be chain I guess, so effectively
any number)
"""
struct HyperMHA{MI,MH}
	mill::MI
	mha::MH 
end

@functor HyperMHA

function HyperMHA(ds::BagNode; graph_dim, graph_layers, dense_dim, dense_layers, heads, head_dims)
	mill = reflectinmodel(ds, d -> ffnn(d, graph_dim, graph_dim, graph_layers), SegmentedMeanMax, fsm=Dict("" => d -> ffnn(d, dense_dim, 1, dense_layers)))
	input_dims = size(mill.im(ds.data),1)
	mha = MultiheadAttention(heads, input_dims, head_dims, input_dims)
	HyperMHA(mill, mha)
end


function (m::HyperMHA)(ds::BagNode)
	x = model.mill.im(ds.data)
	xᵣ = reshape(x, size(x,1), length(ds.bags[1]), length(ds.bags))
	xₕ = model.mha(x)
	xₐ = reshape(xₕ, size(x,1), size(x,2))
	model.mill.bm(model.mill.a(xₐ, ds.bags))
end


function ffnn(idim, hdim, odim, nlayers)
	nlayers == 1 && return(Dense(idim,odim))
	nlayers == 2 && return(Chain(Dense(idim, hdim, relu), Dense(hdim,odim)))
	nlayers == 3 && return(Chain(Dense(idim, hdim, relu), Dense(hdim, hdim, relu), Dense(odim,odim)))
	error("nlayers should be only in [1,3]")
end

function initmodel(ds::AbstractMillNode; graph_dim, graph_layers, dense_dim, dense_layers, kwargs...)
	reflectinmodel(ds, d -> ffnn(d, graph_dim, graph_dim, graph_layers), SegmentedMeanMax, fsm=Dict("" => d -> ffnn(d, dense_dim, 1, dense_layers)))
end

# function initmodel(ds::BagNode; kwargs...)
	# HyperMHA(ds; kwargs...)
# end

function initmodel(ds::MultiGraph;graph_dim, graph_layers, dense_dim, dense_layers, kwargs...)
	MultiModel(ds, graph_dim, graph_layers, d -> ffnn(d, dense_dim, 1, dense_layers))
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
	pddld = HyperExtractor(domain)

	#create model from some problem instance
	model = let 
		problem = load_problem(first(problem_files))
		pddle, state = initproblem(pddld, problem)
		h₀ = pddle(state)
		model = initmodel(h₀; graph_dim, graph_layers, dense_dim, dense_layers, heads, head_dims)
		model
	end


	minibatches = map(train_files) do problem_file
		plan = deserialize(plan_file(problem_file))
		problem = load_problem(problem_file)
		ds = fminibatch(pddld, domain, problem, plan.plan)
		# @set ds.x.data = deduplicate(model.mill.im, ds.x.data)
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
graph_dim = 8
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

filename = joinpath("supervised_hyper1", problem_name, join([loss_name, max_steps,  max_time, graph_layers, graph_dim, dense_layers, dense_dim, seed], "_")*".jls")
experiment(domain_pddl, train_files, problem_files, filename, fminibatch; max_steps, max_time, graph_layers, graph_dim, dense_layers, dense_dim)
