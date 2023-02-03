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
using PDDL: get_facts
using NeuroPlanner: artificial_goal

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

function make_search_tree(domain, problem)
	state = initstate(domain, problem)
	planner = AStarPlanner(HAdd(); max_time=30, save_search = true)
	spec = MinStepsGoal(problem)
	sol = planner(domain, state, spec)
end

function create_training_set_from_tree(pddld, problem_file::AbstractString, fminibatch, n::Int)
	problem = load_problem(problem_file)
	create_training_set_from_tree(pddld, problem, fminibatch, n)
end

function create_training_set_from_tree(pddld, problem::GenericProblem, fminibatch, n::Int)
	domain = pddld.domain
	sol = make_search_tree(domain, problem)
	goal = goalstate(domain, problem)
	st = sol.search_tree
	map(sample(collect(NeuroPlanner.leafs(st)), n, replace = false)) do node_id
		plan, trajectory = SymbolicPlanners.reconstruct(node_id, st)
		trajectory, plan, agoal = artificial_goal(domain, problem, trajectory, plan, goal)
		# linex = LinearExtractor(domain, problem)
		# fminibatch(st, linex, trajectory)
		pddle = NeuroPlanner.add_goalstate(pddld, problem, agoal)
		fminibatch(st, pddle, trajectory)
	end
end

"""
	This is pretty lame and should be done better (one day if it works)
"""
function create_training_set(pddld, problem_file, fminibatch, n, depths; trace_type = :backward)
	problem = load_problem(problem_file)
	domain = pddld.domain
	map(1:n) do _
		depth = rand(depths)
		trajectory, plan, goal = sample_trace(domain, problem, depth, trace_type)
		search_tree = search_tree_from_trajectory(domain, trajectory, plan)
		pddle = NeuroPlanner.add_goalstate(pddld, problem, goal)
		fminibatch(search_tree, pddle, trajectory)
	end
end

		
function sample_trace(domain, problem, depth, trace_type)
	if trace_type == :forward
		for a in 1:tries
			trajectory, plan = sample_forward_trace(domain, problem, depth; remove_cycles = false)
			trajectory, plan, goal = artificial_goal(domain, problem, trajectory, plan)
			!isempty(plan) && return(trajectory, plan, goal)
		end
		error("failed to generate plane from forward sampler")
	end

	if trace_type == :backward
		trajectory, plan = sample_backward_trace(domain, problem, depth)
		goal = goalstate(domain, problem)
		return(trajectory, plan, goal)
	end
	trace_type == :both && return(sample_trace(domain, problem, depth, rand([:forward, :backward])))
	error("either forward_traces or backward_traces has to be true")
end

"""
	This is pretty lame and should be done better (one day if it works)
"""
function create_training_set(pddld, problem_file, fminibatch, n, depths; trace_type = :backward)
	problem = load_problem(problem_file)
	domain = pddld.domain
	map(1:n) do _
		depth = rand(depths)
		trajectory, plan, goal = sample_trace(domain, problem, depth, trace_type)
		search_tree = search_tree_from_trajectory(domain, trajectory, plan)
		pddle = NeuroPlanner.add_goalstate(pddld, problem, goal)
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
			dense_dim = 32, max_epochs = 100, training_set_size = 10000, depths = 1:30, trace_type = :backward)
	isdir(ofile()) || mkpath(ofile())
	domain = load_domain(domain_pddl)
	pddld = PDDLExtractor(domain)

	#create model from some problem instance
	problem = load_problem(problem_file)
	pddle, state = initproblem(pddld, problem)
	h₀ = pddle(state)
	model = MultiModel(h₀, graph_dim, graph_layers, d -> ffnn(d, dense_dim, dense_layers))
	planner = (heuristic; kwargs...) -> ForwardPlanner(;heuristic=heuristic, h_mult=1, kwargs...)
	solve_problem(pddld, problem, model, planner)	#warmup the solver


	# training_set = create_training_set(pddld, problem_file, fminibatch, training_set_size, depths; trace_type)
	training_set = create_training_set_from_tree(pddld, problem_file, fminibatch, training_set_size)
	opt = AdaBelief()
	ps = Flux.params(model)
	fvals = fill(typemax(Float64), length(training_set))
	sol = test_planner(pddld, model, problem_file, planner; max_time = 600)
	stats = map(1:10) do i 
		fvals .= train!(x -> loss_fun(model, x), model, ps, opt, training_set, fvals, max_steps; max_loss, ϵ = epsilon, opt_type = Symbol(opt_type))
		fv = mean(filter(!isinf, fvals))
		solution = test_planner(pddld, model, problem_file, planner; max_time = 600)
		(;i, solution, fv)
	end


	# serialize(filename[end-4:end]*"_model.jls", model) 

	results = []
	for h_mult in [1, 1.5, 2, 2.5]
		planner = (heuristic; kwargs...) -> ForwardPlanner(;heuristic=heuristic, h_mult=h_mult, kwargs...)
		solution = test_planner(pddld, model, problem_file, planner; max_time)
		r = (;h_mult, solution)
		push!(results, r)
		# serialize(filename, results)
	end
end


# Let's make configuration ephemeral
# problem_name = ARGS[1]
problem_name = "gripper"
loss_name = "lstar"
problem_file = "benchmarks/gripper/problems/gripper-n10.pddl"
seed = 1

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
training_set_size = 1000
max_steps = 20000
max_loss = 0.0
depths = 1:30
ttratio = 1.0
trace_type = :forward

Random.seed!(seed)
domain_pddl, problem_files, _ = getproblem(problem_name, false)
loss_fun, fminibatch = NeuroPlanner.getloss(loss_name)
ofile(s...) = joinpath("patterndatabase", problem_name, s...)
filename = ofile(join([loss_name,opt_type,epsilon,max_time,graph_layers,graph_dim,dense_layers,dense_dim,training_set_size,max_steps,max_loss, depths,ttratio,seed], "_")*".jls")
experiment(domain_pddl, problem_files[2], ofile, loss_fun, fminibatch; filename, opt_type, epsilon, max_time, graph_layers, graph_dim, dense_layers, dense_dim, training_set_size, max_steps, depths, max_loss, trace_type)

function parse_results(problem, loss, trajectory_type, trajectory_goal; testset = true)
	namefun(number) = "patterndatabase/$(problem)/$(loss)_$(trajectory_type)_$(trajectory_goal)_20000_30_2_8_2_32_$(number).jls"
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

function show_results(trajectory_goal)
	problems = ["blocks", "ferry", "gripper", "npuzzle"]
	df = map(problems) do problem
		mapreduce(merge,["forward", "backward", "both"]) do t
			parse_results(problem, "lstar", t,trajectory_goal;testset = true)
		end
	end |> DataFrame;
	hcat(DataFrame(problem = problems),  df[:,1:2:end], df[:,2:2:end])
end
