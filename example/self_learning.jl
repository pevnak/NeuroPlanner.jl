using PDDL2Graph
using PDDL
using Flux
using GraphSignals
using GeometricFlux
using SymbolicPlanners
using Statistics
using IterTools
using Random

benchdir(s...) = joinpath("benchmarks","blocks-slaney",s...)
taskfiles(s) = [benchdir(s, f) for f in readdir(benchdir(s)) if startswith(f,"task",)]

function solve_problem(pddld, problem::GenericProblem, model; max_time=30)
	domain = pddld.domain
	pddle, state = PDDL2Graph.initproblem(pddld, problem)
	goal = PDDL.get_goal(problem)
	planner = AStarPlanner(GNNHeuristic(pddld, problem, model); max_time, save_search = true)
	solution_time = @elapsed sol = planner(domain, state, goal)
	sol.status == :success || return(nothing)
	stats = (;solution_time, 
		sol_length = length(sol.trajectory),
		expanded_states = 
		)
	(;sol, stats)
end

solve_problem(pddld, problem_file::AbstractString, model; kwargs...) = solve_problem(pddld, load_problem(problem_file), model;kwargs...)


function update_solutions!(solutions, pddld, model, problem_files; offset = 1, stop_after=32)
	@assert length(solutions) == length(problem_files)
	updated = Int[]
	init_offset = offset
	while true 
		i = offset
		problem = load_problem(problem_files[i])
		newsol = solve_problem(pddld, problem, model)
		oldsol = solutions[i]
		solutions[i] = update_solution(oldsol, newsol, pddld, problem)
		solutions[i] != oldsol && push!(updated, i)
		offset = offset == length(solutions) ? 1 : offset+1
		init_offset == offset && break
		length(updated) ≥ stop_after && break
	end
	offset, updated
end

update_solution(oldsol::Nothing, newsol::Nothing, pddld, problem) = nothing
update_solution(oldsol::NamedTuple, newsol::Nothing, pddld, problem) = oldsol
update_solution(oldsol::Nothing, newsol::PathSearchSolution, pddld, problem) = prepare_minibatch_l2(newsol, pddld, problem)
function update_solution(oldsol::NamedTuple, newsol::PathSearchSolution, pddld, problem) 
	length(oldsol.y) ≤ length(newsol.trajectory) ? oldsol : prepare_minibatch_l2(newsol, pddld, problem)
end
function update_solution(oldsol::NamedTuple, newsol::NamedTuple, pddld, problem) 
	length(oldsol.y) ≤ length(newsol.trajectory) ? oldsol : merge(prepare_minibatch_l2(newsol.sol, pddld, problem), (;stats = newsol.stats))
end
update_solution(oldsol::Nothing, newsol::NamedTuple, pddld, problem) = merge(prepare_minibatch_l2(newsol.sol, pddld, problem), (;stats = newsol.stats))
function prepare_minibatch_l2(sol, pddld, problem)
   pddle, state = PDDL2Graph.initproblem(pddld, problem)
   (;x = PDDL2Graph.sparsegraph(reduce(cat, map(pddle, sol.trajectory))),
     y = collect(length(sol.trajectory):-1:1),
     )
end

function show_stats(solutions)
	solved = filter(!isnothing, solutions)
	mean_time = mean(s.solution_time for s in solved)
	mean_length = mean(length(s.y) for s in solved)
	println(" solved instances: ", length(solved), 
		" (",length(solved) / length(solutions), ") mean length ",mean_length, " mean_time = ", mean_time)
end

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
problem_files = mapreduce(taskfiles, vcat, ["blocks3", "blocks4", "blocks5", "blocks6", "blocks7", "blocks8"])
solutions = Vector{Any}(fill(nothing, length(problem_files)))

#crate model from some problem instance
model = let 
	problem = load_problem(benchdir("blocks3", "task01.pddl"))
	pddle, state = initproblem(pddld, problem)
	h₀ = pddle(state)
	odim_of_graph_conv = 4
	MultiModel(h₀, odim_of_graph_conv, d -> Chain(Dense(d, 32,relu), Dense(32,1)))
end

loss(x, y) = Flux.Losses.mse(vec(model(x)), y)
loss(xy::NamedTuple) = loss(xy.x, xy.y)
loss(xy::Tuple) = loss(xy[1],xy[2])
offset = 1
opt = AdaBelief()
ps = Flux.params(model)

for i in 1:100
	offset, updated = update_solutions!(solutions, pddld, model, problem_files; offset)	
	solved = findall(solutions .!== nothing)
	ii = solved[randperm(length(solved))]
	print("offset = ",offset," ")
	show_stats(solutions)
	#do one epoch on newly solved instances
	t₁ = @elapsed Flux.train!(loss, ps, solutions[updated], opt)
	#do one epoch on all solved instances so far
	t₂ = @elapsed Flux.train!(loss, ps, solutions[ii], opt)
	@show (t₁, t₂)
end