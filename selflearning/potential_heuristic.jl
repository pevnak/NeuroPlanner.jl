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
using DataFrames
using Serialization
using PDDL: get_facts
using NeuroPlanner: artificial_goal

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
struct PotentialHeuristic{P,M} <: Heuristic 
	linex::P
	model::M
end

Base.hash(g::PotentialHeuristic, h::UInt) = hash(g.model, hash(g.linex, h))
SymbolicPlanners.compute(h::PotentialHeuristic, domain::Domain, state::State, spec::Specification) = only(h.model(h.linex(state)))

function solver_stats(sol, solution_time)
	solved = sol.status == :success
	(;solution_time, 
		sol_length = solved ? length(sol.trajectory) : missing,
		expanded = sol.expanded,
		solved,
	)
end 

function solve_problem(linex, problem::GenericProblem, model, init_planner; max_time=30, return_unsolved = false)
	domain = linex.c_domain
	# domain = linex.domain
	state = initstate(domain, problem)
	goal = PDDL.get_goal(problem)
	h = PotentialHeuristic(linex, model)
	# h = EvalTracker(h)
	planner = init_planner(h; max_time, save_search = true)
	solution_time = @elapsed sol = planner(domain, state, goal)
	return_unsolved || sol.status == :success || return(nothing)
	stats = solver_stats(sol, solution_time)
	(;sol, stats)
end


function make_search_tree(domain, problem;max_time = 30)
	state = initstate(domain, problem)
	planner = AStarPlanner(HAdd(); max_time, save_search = true)
	spec = MinStepsGoal(problem)
	solution_time = @elapsed sol = planner(domain, state, spec)
	stats = solver_stats(sol, solution_time)
	(;sol, stats)
end

function create_training_set_from_tree(linex, problem::GenericProblem, fminibatch, n::Int)
	domain = linex.domain
	sol, stats = make_search_tree(domain, problem)
	goal = goalstate(linex.domain, problem)
	st = sol.search_tree
	goal_states = collect(NeuroPlanner.leafs(st))
	goal_states = length(goal_states) > n ? sample(goal_states, n, replace = false) : goal_states
	minibatches = map(goal_states) do node_id
		plan, trajectory = SymbolicPlanners.reconstruct(node_id, st)
		trajectory, plan, agoal = artificial_goal(domain, problem, trajectory, plan, goal)
		lx = NeuroPlanner.add_goalstate(linex, agoal)
		fminibatch(st, lx, trajectory)
	end
	minibatches, stats
end

# Let's make configuration ephemeral
# problem_name = "agricola-sat18"
# problem_name = "caldera-sat18"
# problem_name = "woodworking-sat11-strips"
# problem_name = "gripper"
# problem_name = "blocks"
# loss_name = "lgbfs"
# loss_name = "lstar"
# loss_name = "lrt"
# seed = 1

problem_name = ARGS[1]
loss_name = ARGS[2]
seed = parse(Int,ARGS[3])


opt_type = :mean
# opt_type = :worst
epsilon = 0.5
max_time = 30
dense_layers = 1
dense_dim = 32
training_set_size = 1000
max_steps = 10000
max_loss = 0.0

Random.seed!(seed)
domain_pddl, problem_files, _ = getproblem(problem_name, false)
loss_fun, fminibatch = NeuroPlanner.getloss(loss_name)
ofile(s...) = joinpath("potential", problem_name, s...)
filename = ofile(join([loss_name, opt_type ,epsilon ,max_time ,dense_layers ,dense_dim ,training_set_size ,max_steps ,max_loss, seed], "_")*".jls")
!isdir(ofile()) && mkpath(ofile())

@show (problem_name, loss_name, seed)
@show filename


results = DataFrame()
problem_file = problem_files[1]
# problem_file = "benchmarks/gripper/problems/gripper-n50.pddl"
for problem_file in problem_files
	@show problem_file
	domain = load_domain(domain_pddl)
	problem = load_problem(problem_file)
	c_domain, c_state = compiled(domain, problem)
	linex = LinearExtractor(domain, problem, c_domain, c_state)

	#create model from some problem instance
	h₀ = linex(c_state)
	model = ffnn(length(h₀), dense_dim, dense_layers)
	init_planner = (heuristic; kwargs...) -> ForwardPlanner(;heuristic=heuristic, h_mult=1, kwargs...)
	sol = solve_problem(linex, problem, model, init_planner; max_time = 1)	#warmup the solver

	# training_set = create_training_set(pddld, problem_file, fminibatch, training_set_size, depths; trace_type)
	time_trainset = @elapsed training_set, train_stats = create_training_set_from_tree(linex, problem, fminibatch, training_set_size)
	opt = AdaBelief()
	ps = Flux.params(model)
	fvals = fill(typemax(Float64), length(training_set))
	ttrain = @elapsed train!(x -> loss_fun(model, x), model, ps, opt, training_set, fvals, max_steps; max_loss, ϵ = epsilon, opt_type = Symbol(opt_type))
	tset = @elapsed test_stats = solve_problem(linex, problem, model, init_planner; max_time, return_unsolved = true).stats
	push!(results, (;
		problem_file,
		test_stats...,
		train_stats,
		ttrain,
		time_trainset,
		tset,
		fval = mean(filter(!isnan, fvals))
		), promote = true)

	display(results)
	serialize(filename, results)
end


function parse_results(problem, loss, layers; testset = true)
	namefun(number) = "potential/$(problem)/$(loss)_mean_0.5_30_$(layers)_32_1000_10000_0.0_$(number).jls"
	n = (Symbol(loss*"_pot"), Symbol(loss*"_Hadd"))
	numbers = filter(isfile ∘ namefun, 1:3)
	isempty(numbers) && return(NamedTuple{n}((NaN,NaN)))
	x = map(numbers) do number
		stats = deserialize(namefun(number))
		[mean(stats.solved), mean(s.solved for s in stats.train_stats)]
	end |> mean
	x = round.(x, digits = 2)
	NamedTuple{n}(tuple(x...))
end

function show_results(layers)
	problems = ["blocks", "ferry", "gripper", "npuzzle"]
	df = map(problems) do problem
		mapreduce(merge,["lstar", "lrt","l2"]) do l
			parse_results(problem, l, layers)
		end
	end |> DataFrame;
	hcat(DataFrame(problems = problems), df[:,[1,3,5,6]])
end

for (problem, l, number) in Iterators.product(["blocks", "ferry", "gripper", "npuzzle"], 1:2, 1:3)
	s = "potential/$(problem)/lstar_mean_0.5_30_$(l)_32_1000_10000_0.0_$(number).jls"
	d = "potential/$(problem)/lstar2_mean_0.5_30_$(l)_32_1000_10000_0.0_$(number).jls"
	run(`mv $s $d`)
end
