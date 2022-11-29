using PDDL2Graph
using PDDL
using Flux
using GraphSignals
using GeometricFlux
using SymbolicPlanners
using PDDL: GenericProblem, PathSearchSolution
using Statistics
using IterTools
using Random
using StatsBase
using Serialization

include("solution_tracking.jl")
include("problems.jl")
include("losses.jl")

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

function experiment(domain_pddl, problem_files, ofile, loss_fun, fminibatch; solve_solved = false, stop_after=32)
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
		solve_problem(pddld, problem, model)	#warmup the solver
		model
	end

	offset = 1
	opt = AdaBelief()
	ps = Flux.params(model)
	all_solutions = []
	losses = []
	for i in 1:10
		offset = 1
		while offset < length(solutions)
			offset, updated = update_solutions!(solutions, pddld, model, problem_files, fminibatch; offset, solve_solved , stop_after)	
			solved = findall(solutions .!== nothing)
			print("offset = ", offset," updated = ", length(updated), " ")
			show_stats(solutions)
			length(solved) == length(solutions) && break
			#do one epoch on newly solved instances
			updated_solutions = [s.minibatch for s in solutions[updated] if nonempty(s.minibatch)];
			t₁ = @elapsed length(updated) > 0 && Flux.train!(x -> loss_fun(model, x), ps, updated_solutions, opt)
			#do one epoch on all solved instances so far
			# t₂ = @elapsed Flux.train!(loss, ps, solutions[ii], opt)		
			#do one epoch on all solved instances but prioriteze those with the largest number of expanded nodes

			# we should actually 
			solved = filter(nonempty, solutions[solved]);
			w = StatsBase.Weights([s.stats.expanded for s in solved]);
			mbs = [s.minibatch for s in solved]
			t₂ = @elapsed Flux.train!(x -> loss_fun(model, x), ps, repeatedly(() -> sample(mbs, w), 1000), opt)
			@show (t₁, t₂)
		end
		l = [loss_fun(model, s) for s in solutions if s !== nothing && nonempty(s)]
		push!(losses, l)
		println("loss after $(i) epoch = ", mean(l))
		push!(all_solutions, [(s == nothing ? nothing : s.stats) for s in solutions])
		serialize(ofile("$(loss_name)_$(solve_solved)_$(stop_after)_$(seed).jls"),(;all_solutions, losses))
		all(s !== nothing for s in solutions) && break
	end
end

# problem_name = ARGS[1]
# loss_name = ARGS[2]
# seed = parse(Int, ARGS[3])

problem_name = "blocks"
loss_name = "l2"
seed = 1

Random.seed!(seed)
domain_pddl, problem_files, ofile = getproblem(problem_name)
loss_fun, fminibatch = get_loss(loss_name)
experiment(domain_pddl, problem_files, ofile, loss_fun, fminibatch)
