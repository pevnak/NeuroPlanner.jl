using PDDL2Graph
using PDDL
using Flux
using GraphSignals
using GeometricFlux
using SymbolicPlanners
using Statistics
using IterTools
using Random
using StatsBase
using Serialization

benchdir(s...) = joinpath("benchmarks","blocks-slaney",s...)
taskfiles(s) = [benchdir(s, f) for f in readdir(benchdir(s)) if startswith(f,"task",)]
include("solution_tracking.jl")
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

create_example(pddld, filename::String) = create_example(pddld, load_problem(filename))	
create_examples(pddld, dname::String) = [create_example(pddld, f) for f in taskfiles(dname)]

domain = load_domain(benchdir("domain.pddl"))
pddld = PDDLExtractor(domain)
problem_files = mapreduce(taskfiles, vcat, ["blocks$(i)" for i in 3:15])
solutions = Vector{Any}(fill(nothing, length(problem_files)))

#create model from some problem instance
model = let 
	problem = load_problem(benchdir("blocks10", "task01.pddl"))
	pddle, state = initproblem(pddld, problem)
	h₀ = pddle(state)
	odim_of_graph_conv = 8
	model = MultiModel(h₀, odim_of_graph_conv, d -> Chain(Dense(d, 32,relu), Dense(32,1)))
	solve_problem(pddld, problem, model)	#warmup the solver
	model
end

# l₂loss(x, y) = Flux.Losses.mse(vec(model(x)), y)
# l₂loss(xy::NamedTuple) = l₂loss(xy.x, xy.y)
# l₂loss(xy::Tuple) = l₂loss(xy[1],xy[2])
# fminibatch = prepare_minibatch_l2
"""
lₛloss(x, g, H₊, H₋)

Minimizes `L*` loss, We want ``f * H₋ .< f * H₊``, which means to minimize cases when ``f * H₋ .> f * H₊``
"""
function lₛloss(x, g, H₊, H₋)
	g = reshape(g, 1, :)
	f = model(x) .+ g
	mean(softplus.(f * H₋ .- f * H₊))
end
lₛloss(xy::NamedTuple) = lₛloss(xy.x, xy.path_cost, xy.H₊, xy.H₋)
nonempty(s::NamedTuple) = !isempty(s.H₊) && !isempty(s.H₋)
fminibatch = prepare_minibatch_lₛ
loss = lₛloss
offset = 1
opt = AdaBelief()
ps = Flux.params(model)
all_solutions = []
losses = []
for i in 1:10
	offset = 1
	while offset < length(solutions)
		offset, updated = update_solutions!(solutions, pddld, model, problem_files, fminibatch; offset, solve_solved = true, stop_after=32)	
		solved = findall(solutions .!== nothing)
		print("offset = ", offset," updated = ", length(updated), " ")
		show_stats(solutions)
		#do one epoch on newly solved instances
		updated_solutions = filter(nonempty, solutions[updated])
		t₁ = @elapsed length(updated) > 0 && Flux.train!(loss, ps, updated_solutions, opt)
		#do one epoch on all solved instances so far
		# t₂ = @elapsed Flux.train!(loss, ps, solutions[ii], opt)		
		#do one epoch on all solved instances but prioriteze those with the largest number of expanded nodes
		solved = solutions[solved];
		solved = filter(nonempty, solved);

		# we should actually 
		w = StatsBase.Weights([s.stats.expanded for s in solved]);
		t₂ = @elapsed Flux.train!(loss, ps, repeatedly(() -> sample(solved, w), 1000), opt)
		@show (t₁, t₂)
	end
	l = [loss(s) for s in solutions if s !== nothing && nonempty(s)]
	push!(losses, l)
	println("loss after $(i) epoch = ", mean(l))
	push!(all_solutions, [(s == nothing ? nothing : s.stats) for s in solutions])
	serialize("stats_ls.jls",(;all_solutions, losses))
end
