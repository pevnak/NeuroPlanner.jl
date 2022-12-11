using NeuroPlanner
using PDDL
using Flux
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

# problem_name = "ferry"
# loss_name = "l2"
# seed = 1

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

function experiment(domain_pddl, problem_files, ofile, loss_fun, fminibatch, planner; opt_type = :worst, solve_solved = false, stop_after=32, filename = "", max_time=30, double_maxtime=false, max_steps = 100, max_loss = 0.0, epsilon = 0.5, artificial_goals = false)
	isdir(ofile()) || mkpath(ofile())
	domain = load_domain(domain_pddl)
	pddld = PDDLExtractor(domain)
	solutions = Vector{Any}(fill(nothing, length(problem_files)))

	#create model from some problem instance
	model = let 
		problem = load_problem(first(problem_files))
		pddle, state = initproblem(pddld, problem)
		h₀ = pddle(state)
		odim_of_graph_conv = 8
		model = MultiModel(h₀, odim_of_graph_conv, d -> Chain(Dense(d, 32,relu), Dense(32,1)))
		solve_problem(pddld, problem, model, planner)	#warmup the solver
		model
	end

	offset = 1
	opt = AdaBelief()
	ps = Flux.params(model)
	all_solutions = []
	losses = []
	fvals = fill(typemax(Float64), length(solutions))
	for epoch in 1:10
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
			t₁ = @elapsed fvals[ii] .= train!(x -> loss_fun(model, x), ps, opt, solutions[ii], fvals[ii], max_steps; max_loss, ϵ = epsilon, opt_type)
			l = filter(x -> x != typemax(Float64), fvals)
			println("epoch = $(epoch) offset $(offset)  mean error = ", mean(l), " worst case = ", maximum(l)," time: = ", t₁)
		end
		solved_after = issolved.(solutions)
		push!(losses, deepcopy(fvals))
		push!(all_solutions, filter(issolved, solutions))
		if !isempty(filename)
			serialize(filename,(;all_solutions, losses))
		end
		all(issolved.(solutions)) && break
		if sum(solved_after) == sum(solved_before)  && double_maxtime
			max_time *= 2
		end
	end
	all_solutions
end

# problem_name = ARGS[1]
# loss_name = ARGS[2]
# seed = parse(Int, ARGS[3])

problem_name = "blocks"
seed = 1
double_maxtime = false
# loss_name = "lgbfs"
# planner_name = "gbfs"
loss_name = "lstar"
planner_name = "astar"
solve_solved = false
stop_after = 32
max_steps = 2000
# opt_type = :worst
opt_type = :mean
epsilon = 0.5
max_loss = 0.0
max_time = 30
sort_by_complexity = true
artificial_goals = true

Random.seed!(seed)
domain_pddl, problem_files, ofile = getproblem(problem_name, sort_by_complexity)
planner = (planner_name == "astar") ? AStarPlanner : GreedyPlanner
loss_fun, fminibatch = getloss(loss_name)
filename = ofile("$(planner_name)_$(loss_name)_$(solve_solved)_$(stop_after)_$(seed).jls")
problem_files = problem_files[end-20:end]
experiment(domain_pddl, problem_files, ofile, loss_fun, fminibatch, planner; double_maxtime, filename, solve_solved, stop_after, max_steps, max_loss, max_time, opt_type, epsilon, artificial_goals)
