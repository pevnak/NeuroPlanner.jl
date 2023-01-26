#  This script is meant to understand, why learning from backward 
#  traces does not work well, but learning from forward traces does.
#  My suspicion us that it is cause forward traces contains a full view
#  of the state, whereas traces does not. The idea is therefore to take 
#  solved plans and create forward and reverse plans from them and compare
#  the results. If the assumption is correct, there forward plans should be much 
#  better than the reverse plans.
#

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
using NeuroPlanner: execute_backward_plan

include("solution_tracking.jl")
include("problems.jl")
include("training.jl")

function ffnn(d, odim, nlayers)
	nlayers == 1 && return(Dense(d,1))
	nlayers == 2 && return(Chain(Dense(d, odim, relu), Dense(odim,1)))
	nlayers == 3 && return(Chain(Dense(d, odim, relu), Dense(odim, odim, relu), Dense(odim,1)))
	error("nlayers should be only in [1,3]")
end

function plan_file(problem_file)
	middle_path  = splitpath(problem_file)[3:end-1]
	middle_path = filter(∉(("problems",)),middle_path)
	middle_path = filter(∉(("test",)),middle_path)
	joinpath("plans", problem_name, middle_path..., basename(problem_file)[1:end-5]*".jls")
end

function create_minibatch(pddld, domain, problem, plan, fminibatch, trajectory_type, trajectory_goal; verifyplan = false)
	samples = []
	verifyplan && NeuroPlanner.verify_plan(domain, problem, plan)
	if trajectory_type ∈ [:forward, :both]
		trajectory = SymbolicPlanners.simulate(StateRecorder(), domain, initstate(domain, problem), plan)
		goal = trajectory_goal ? trajectory[end] : goalstate(domain, problem)
		pddle = NeuroPlanner.add_goalstate(pddld, problem, goal)
		push!(samples, fminibatch(pddle, domain, problem, trajectory; goal_aware = false))
	end
	if trajectory_type ∈ [:backward, :both]
		trajectory = execute_backward_plan(domain, problem, reverse(plan))[1]
		goal = trajectory_goal ? trajectory[end] : goalstate(domain, problem)
		pddle = NeuroPlanner.add_goalstate(pddld, problem, goal)
		push!(samples, fminibatch(pddle, domain, problem, trajectory; goal_aware = false))
	end
	samples
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

function experiment(domain_pddl, train_files, problem_files, filename, loss_fun, fminibatch;max_steps = 10000, max_time = 30, graph_layers = 2, graph_dim = 8, dense_layers = 2, dense_dim = 32, trajectory_type = :forward, trajectory_goal = false)
	!isdir(dirname(filename)) && mkpath(dirname(filename))
	domain = load_domain(domain_pddl)
	pddld = PDDLExtractor(domain)

	#create model from some problem instance
	model = let 
		problem = load_problem(first(problem_files))
		pddle, state = initproblem(pddld, problem)
		h₀ = pddle(state)
		model = MultiModel(h₀, graph_dim, graph_layers, d -> ffnn(d, dense_dim, dense_layers))
		model
	end


	minibatches = mapreduce(vcat, train_files) do problem_file
		plan = deserialize(plan_file(problem_file))
		problem = load_problem(problem_file)
		create_minibatch(pddld, domain, problem, plan.plan, fminibatch, trajectory_type, trajectory_goal; verifyplan = true)
	end

	opt = AdaBelief()
	ps = Flux.params(model)
	train!(x -> loss_fun(model, x), model, ps, opt, () -> rand(minibatches), max_steps)
	serialize(filename[1:end-4]*"_model.jls", model)	


	stats = map(Iterators.product([AStarPlanner, GreedyPlanner], problem_files)) do (planner, problem_file)
		used_in_train = problem_file ∈ train_files
		@show problem_file
		sol = solve_problem(pddld, problem_file, model, planner;return_unsolved = true)
		trajectory = sol.sol.status == :max_time ? nothing : sol.sol.trajectory
		merge(sol.stats, (;used_in_train, planner = "$(planner)", trajectory))
	end
	serialize(filename, stats)
end

# Let's make configuration ephemeral
problem_name = ARGS[1]
# loss_name = ARGS[2]
trajectory_type = Symbol(ARGS[2])
seed = parse(Int, ARGS[3])

# problem_name = "ferry"
# problem_name = "npuzzle"
# problem_name = "blocks"
# problem_name = "gripper"
# seed = 1
# trajectory_type = :both

loss_name = "lstar"
max_steps = 20000
max_time = 30
graph_layers = 2
graph_dim = 8
dense_layers = 2
dense_dim = 32
trajectory_goal = false

Random.seed!(seed)
domain_pddl, problem_files, ofile = getproblem(problem_name, false)
train_files = filter(isfile ∘ plan_file, problem_files)
train_files = sample(train_files, min(length(train_files), div(length(problem_files), 2)), replace = false)
loss_fun, fminibatch = getloss(loss_name)

filename = joinpath("investigate_regression", problem_name, join([loss_name, trajectory_type, trajectory_goal, max_steps,  max_time, graph_layers, graph_dim, dense_layers, dense_dim, seed], "_")*".jls")
experiment(domain_pddl, train_files, problem_files, filename, loss_fun, fminibatch; max_steps, max_time, graph_layers, graph_dim, dense_layers, dense_dim, trajectory_type, trajectory_goal)


function parse_results(problem, loss, trajectory_type; testset = true)
	namefun(number) = "investigate_regression/$(problem)/$(loss)_$(trajectory_type)_true_20000_30_2_8_2_32_$(number).jls"
	n = (Symbol("$(trajectory_type)_astar"), Symbol("$(trajectory_type)_gbfs"))
	numbers = filter(isfile ∘ namefun, 1:3)
	isempty(numbers) && return(NamedTuple{n}((NaN,NaN)))
	x = map(numbers) do number
		filename = namefun(number)
		stats = deserialize(filename)
		df = DataFrame(stats[:])
		da = filter(df) do r 
			(r.used_in_train !== testset) && (r.planner == "AStarPlanner")
		end
		dg = filter(df) do r 
			(r.used_in_train !== testset) && (r.planner == "GreedyPlanner")
		end
		[mean(da.solved), mean(dg.solved)]
	end |> mean
	x = round.(x, digits = 2)
	NamedTuple{n}(tuple(x...))
end

function show_results()
	problems = ["blocks", "ferry", "gripper", "npuzzle"]
	df = map(problems) do problem
		mapreduce(merge,["forward", "backward", "both"]) do t
			parse_results(problem, "lstar", t;testset = true)
		end
	end |> DataFrame;
	hcat(DataFrame(problem = problems),  df[:,1:2:end], df[:,2:2:end])
end
