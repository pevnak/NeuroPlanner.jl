using PDDL2Graph
using PDDL
using Flux
using GraphSignals
using GeometricFlux
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

GNNHeuristic(pddld, problem, model) = GNNHeuristic(PDDL2Graph.add_goalstate(pddld, problem), model)
Base.hash(g::GNNHeuristic, h::UInt) = hash(g.model, hash(g.pddle, h))
SymbolicPlanners.compute(h::GNNHeuristic, domain::Domain, state::State, spec::Specification) = only(h.model(h.pddle(state)))

function experiment(domain_pddl, problem_files, ofile, loss_fun, fminibatch, planner; opt_type = :worst, solve_solved = false, stop_after=32, filename = "", max_time=30, double_maxtime=false, max_steps = 100, max_loss = 0.0, epsilon = 0.5)
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
	for i in 1:10
		solved_before = solutions .!== nothing
		offset = 1
		while offset < length(solutions)
			offset, updated = update_solutions!(solutions, pddld, model, problem_files, fminibatch, planner; offset, solve_solved, stop_after, max_time)	
			# print some Statistics
			solved = findall(solutions .!== nothing)
			print("offset = ", offset," updated = ", length(updated), " ")
			show_stats(solutions)
			length(solved) == length(solutions) && break
			
			t₁ = @elapsed fvals[solved] .= train!(x -> loss_fun(model, x.minibatch), ps, opt, solutions[solved], fvals[solved], max_steps; max_loss, ϵ = epsilon, opt_type)
		end
		solved_after = solutions .!== nothing
		l = filter(x -> x !== typemax(Float64), fvals)
		push!(losses, l)
		println("loss after $(i) epoch = ", mean(l), " max time = ", max_time)
		push!(all_solutions, [(s == nothing ? nothing : s.stats) for s in solutions])
		if !isempty(filename)
			serialize(filename,(;all_solutions, losses))
		end
		all(s !== nothing for s in solutions) && break
		if sum(solved_after) == sum(solved_before) 
			if double_maxtime
				max_time *= 2
			else 
				break
			end
		end
	end
	all_solutions
end

# problem_name = ARGS[1]
# loss_name = ARGS[2]
# seed = parse(Int, ARGS[3])

problem_name = "blocks"
# loss_name = "lgbfs"
loss_name = "lstar"
seed = 1
double_maxtime = false
planner_name = "astar"
# planner_name = "gbfs"
solve_solved = false
stop_after = 32
max_steps = 100
opt_type = :worst
epsilon = 0.5
max_loss = 0.0
max_time = 30
sort_by_complexity = true

Random.seed!(seed)
domain_pddl, problem_files, ofile = getproblem(problem_name, sort_by_complexity)
planner = (planner_name == "astar") ? AStarPlanner : GreedyPlanner
loss_fun, fminibatch = getloss(loss_name)
filename = ofile("$(planner_name)_$(loss_name)_$(solve_solved)_$(stop_after)_$(seed).jls")
experiment(domain_pddl, problem_files, ofile, loss_fun, fminibatch, planner; double_maxtime, filename, solve_solved, stop_after, max_steps, max_loss, max_time, opt_type, epsilon)
