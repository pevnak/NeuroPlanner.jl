using Term
using SparseArrays
using OneHotArrays
verbose::Bool = true

function solve_problem(pddld, problem::GenericProblem, model; max_time=30)
	domain = pddld.domain
	pddle, state = PDDL2Graph.initproblem(pddld, problem)
	goal = PDDL.get_goal(problem)
	planner = AStarPlanner(GNNHeuristic(pddld, problem, model); max_time, save_search = true)
	solution_time = @elapsed sol = planner(domain, state, goal)
	sol.status == :success || return(nothing)
	stats = (;solution_time, 
		sol_length = length(sol.trajectory),
		expanded = sol.expanded,
		)
	(;sol, stats)
end

solve_problem(pddld, problem_file::AbstractString, model; kwargs...) = solve_problem(pddld, load_problem(problem_file), model;kwargs...)


function update_solutions!(solutions, pddld, model, problem_files, fminibatch; offset = 1, stop_after=32, cycle = false, solve_solved = false)
	@assert length(solutions) == length(problem_files)
	updated_solutions = Int[]
	init_offset = offset
	while true 
		i = offset
		problem = load_problem(problem_files[i])
		oldsol = solutions[i]
		if solve_solved || oldsol == nothing
			newsol = solve_problem(pddld, problem, model)
			solutions[i] = update_solution(oldsol, newsol, pddld, problem, fminibatch)
			solutions[i] != oldsol && push!(updated_solutions, i)
		end
		!cycle && offset == length(solutions) && break 
		offset = offset == length(solutions) ? 1 : offset+1
		init_offset == offset && break
		length(updated_solutions) ≥ stop_after && break
	end
	offset, updated_solutions
end

update_solution(oldsol::Nothing, newsol::Nothing, pddld, problem, fminibatch) = nothing
update_solution(oldsol::Nothing, newsol::PathSearchSolution, pddld, problem, fminibatch) = fminibatch(newsol, pddld, problem, fminibatch)
function update_solution(oldsol::NamedTuple, newsol::PathSearchSolution, pddld, problem, fminibatch) 
	length(oldsol.stats.sol_length) ≤ length(newsol.trajectory) ? oldsol : fminibatch(newsol, pddld, problem, fminibatch)
end

function update_solution(oldsol::NamedTuple, newsol::NamedTuple, pddld, problem, fminibatch)
	o, n = oldsol.stats.sol_length,  newsol.stats.sol_length
	if o < n
		print(@red "longer: $(o) -> $(n)  ")
	elseif o > n
		print(@green "shorter: $(o) -> $(n)  ")
	else
		print(@yellow "same length: $(o) -> $(n)  ")
	end

	o, n = oldsol.stats.expanded,  newsol.stats.expanded
	if o < n
		println(@red "more states: $(o) -> $(n)")
	elseif o > n
		println(@green "less states: $(o) -> $(n)")
	else
		println(@yellow "same states: $(o) -> $(n)")
	end
	stats = (;solution_time = newsol.stats.solution_time,
		sol_length = min(oldsol.stats.sol_length, newsol.stats.sol_length),
		expanded = newsol.stats.expanded,
		)
	length(oldsol.stats.sol_length) ≤ length(newsol.sol.trajectory) ? oldsol : merge(fminibatch(newsol.sol, pddld, problem), (;stats))
end

function update_solution(oldsol::Nothing, newsol::NamedTuple, pddld, problem, fminibatch) 
	verbose && println(@green "new solution: $((newsol.stats.sol_length, newsol.stats.expanded))")
	merge(fminibatch(newsol.sol, pddld, problem), (;stats = newsol.stats))
end

function update_solution(oldsol::NamedTuple, newsol::Nothing, pddld, problem, fminibatch)
	verbose && println(@red "new is unsolved: $((oldsol.stats.sol_length, oldsol.stats.expanded))")
	# stats = (;solution_time = oldsol.stats.solution_time,
	# sol_length = oldsol.stats.sol_length,
	# expanded = 10*oldsol.stats.expanded, # let's try to increase priority of this poor guy
	# )
	oldsol
end

function prepare_minibatch_l2(sol, pddld, problem)
   pddle, state = PDDL2Graph.initproblem(pddld, problem)
   (;x = PDDL2Graph.sparsegraph(reduce(cat, map(pddle, sol.trajectory))),
     y = collect(length(sol.trajectory):-1:1),
     )
end

function prepare_minibatch_lₛ(sol, pddld, problem)
	sol.search_tree === nothing && error("solve the problem with `save_search=true` to keep the search tree")
	pddle, state = PDDL2Graph.initproblem(pddld, problem)

	# get indexes of the states on the solution path, which seems to be hashes of states 
	trajectory_id = hash.(sol.trajectory)
	child₁ = off_path_childs(sol, trajectory_id)
	# child₂ = off_path_childs(sol, union(child₁, trajectory_id))
	ids = vcat(trajectory_id, child₁)
	path_cost = [sol.search_tree[i].path_cost for i in ids]
	states = [sol.search_tree[i].state for i in ids]
	# we want every state on the solution path to be smaller than  
	pm = [(i,j) for i in 1:length(trajectory_id) for j in length(trajectory_id)+1:length(ids)]
	H₊ = onehotbatch([i[2] for i in pm], 1:length(ids))
	H₋ = onehotbatch([i[1] for i in pm], 1:length(ids))

	(;x = PDDL2Graph.sparsegraph(reduce(cat, map(pddle, states))),
         sol_length = length(trajectory_id), H₊, H₋, path_cost)
end

off_path_childs(sol, parents_id::Vector) = off_path_childs(sol, Set(parents_id))

function off_path_childs(sol, parents_id::Set)
	childs = [s.id for s in values(sol.search_tree) if s.parent_id ∈ parents_id]
	setdiff(childs, parents_id)
end

function show_stats(solutions)
	solved = filter(!isnothing, solutions)
	stats = [s.stats for s in solved]
	mean_time = round(mean(s.solution_time for s in stats), digits = 2)
	mean_length = round(mean(s.sol_length for s in stats), digits = 2)
	mean_expanded = round(mean(s.expanded for s in stats), digits = 2)
	println(" solved instances: ", length(solved), 
		" (",round(length(solved) / length(solutions), digits = 2), ") mean length ",mean_length, " mean expanded ",mean_expanded, " mean_time = ", round(mean_time, digits = 3))
end

function _show_stats(stats)
	solved = filter(!isnothing, stats)
	mean_time = round(mean(s.solution_time for s in solved), digits = 2)
	mean_length = round(mean(s.sol_length for s in solved), digits = 2)
	mean_expanded = round(mean(s.expanded for s in solved), digits = 2)
	mean_excess = round(mean(s.expanded ./ s.sol_length for s in solved), digits = 2)
	println(" solved instances: ", length(solved), 
		" (",round(length(solved) / length(stats), digits = 2), ") mean length ",mean_length, " mean expanded ",mean_expanded, " mean expanded excess ",mean_excess, " mean_time = ", round(mean_time, digits = 3))
end
