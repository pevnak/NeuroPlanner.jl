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

function solve_problem(linex, problem::GenericProblem, model, init_planner; max_time=30, return_unsolved = false)
	domain = linex.c_domain
	state = initstate(domain, problem)
	goal = PDDL.get_goal(problem)
	planner = init_planner(PotentialHeuristic(linex, model); max_time, save_search = true)
	solution_time = @elapsed sol = planner(domain, state, goal)
	solved = sol.status == :success
	return_unsolved || solved || return(nothing)
	stats = (;solution_time, 
		sol_length = solved ? length(sol.trajectory) : missing,
		expanded = sol.expanded,
		solved,
		)
	(;sol, stats)
end

function make_search_tree(domain, problem;max_time = 30)
	state = initstate(domain, problem)
	planner = AStarPlanner(HAdd(); max_time, save_search = true)
	spec = MinStepsGoal(problem)
	sol = planner(domain, state, spec)
end

function create_training_set_from_tree(linex, problem::GenericProblem, fminibatch, n::Int)
	domain = linex.domain
	sol = make_search_tree(domain, problem)
	goal = goalstate(linex.domain, problem)
	st = sol.search_tree
	map(sample(collect(NeuroPlanner.leafs(st)), n, replace = false)) do node_id
		plan, trajectory = SymbolicPlanners.reconstruct(node_id, st)
		trajectory, plan, agoal = artificial_goal(domain, problem, trajectory, plan, goal)
		lx = NeuroPlanner.add_goalstate(linex, agoal)
		fminibatch(st, lx, trajectory)
	end
end

# Let's make configuration ephemeral
# problem_name = ARGS[1]
problem_name = "gripper"
loss_name = "lstar"
problem_file = "benchmarks/gripper/problems/gripper-n10.pddl"
problem_file = "benchmarks/gripper/problems/gripper-n47.pddl"
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
max_steps = 10000
max_loss = 0.0
depths = 1:30
ttratio = 1.0
trace_type = :forward

Random.seed!(seed)
domain_pddl, problem_files, _ = getproblem(problem_name, false)
loss_fun, fminibatch = NeuroPlanner.getloss(loss_name)
ofile(s...) = joinpath("potential", problem_name, s...)
filename = ofile(join([loss_name,opt_type,epsilon,max_time,graph_layers,graph_dim,dense_layers,dense_dim,training_set_size,max_steps,max_loss, depths,ttratio,seed], "_")*".jls")

domain = load_domain(domain_pddl)
problem = load_problem(problem_file)
c_domain, c_state = compiled(domain, problem)
linex = LinearExtractor(domain, problem, c_domain, c_state)

#create model from some problem instance
h₀ = linex(c_state)
model = ffnn(length(h₀), dense_dim, dense_layers)
init_planner = (heuristic; kwargs...) -> ForwardPlanner(;heuristic=heuristic, h_mult=1, kwargs...)
sol = solve_problem(linex, problem, model, init_planner)	#warmup the solver


# training_set = create_training_set(pddld, problem_file, fminibatch, training_set_size, depths; trace_type)
tset = @elapsed training_set = create_training_set_from_tree(linex, problem, fminibatch, training_set_size)
opt = AdaBelief()
ps = Flux.params(model)
fvals = fill(typemax(Float64), length(training_set))
ttrain = @elapsed train!(x -> loss_fun(model, x), model, ps, opt, training_set, fvals, max_steps; max_loss, ϵ = epsilon, opt_type = Symbol(opt_type))
tset = @elapsed sol = solve_problem(linex, problem, model, init_planner; max_time=600, return_unsolved = true).stats
