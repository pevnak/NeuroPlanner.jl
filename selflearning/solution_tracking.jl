using Term
using SparseArrays
using OneHotArrays
verbose::Bool = true

function solve_problem(pddld, problem::GenericProblem, model, init_planner; max_time=30, return_unsolved = false)
	domain = pddld.domain
	pddle, state = NeuroPlanner.initproblem(pddld, problem)
	goal = PDDL.get_goal(problem)
	planner = init_planner(GNNHeuristic(pddld, problem, model); max_time, save_search = true)
	solution_time = @elapsed sol = planner(domain, state, goal)
	return_unsolved || sol.status == :success || return(nothing)
	stats = (;solution_time, 
		sol_length = length(sol.trajectory),
		expanded = sol.expanded,
		solved = sol.status == :success
		)
	(;sol, stats)
end

solve_problem(pddld, problem_file::AbstractString, model, init_planner; kwargs...) = solve_problem(pddld, load_problem(problem_file), model, init_planner; kwargs...)

function update_solutions!(solutions, pddld, model, problem_files, fminibatch, planner; offset = 1, stop_after=32, cycle = false, solve_solved = false, max_time = 30, artificial_goals = false)
	@assert length(solutions) == length(problem_files)
	updated_solutions = Int[]
	init_offset = offset
	while true 
		i = offset
		problem = load_problem(problem_files[i])
		oldsol = solutions[i]
		if solve_solved || !issolved(oldsol)
			newsol = solve_problem(pddld, problem, model, planner; max_time, return_unsolved = artificial_goals)
			solutions[i] = update_solution(oldsol, newsol, pddld, problem, fminibatch)
			solutions[i] != oldsol && push!(updated_solutions, i)
		end
		!cycle && offset == length(solutions) && break 
		offset = (offset == length(solutions)) ? 1 : offset+1
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
	!oldsol.stats.solved && return(update_solution(nothing, newsol, pddld, problem, fminibatch))
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
	length(oldsol.stats.sol_length) ≤ length(newsol.sol.trajectory) ? oldsol : (;minibatch = fminibatch(newsol.sol, pddld, problem), stats)
end

function update_solution(oldsol::Nothing, newsol::NamedTuple, pddld, problem, fminibatch) 
	verbose && newsol.stats.solved && println(@green "new solution: $((newsol.stats.sol_length, newsol.stats.expanded))")
	verbose && !newsol.stats.solved && println(@yellow "not solved: $((newsol.stats.sol_length, newsol.stats.expanded))")
	(;minibatch = fminibatch(newsol.sol, pddld, problem), stats = newsol.stats)
end

function update_solution(oldsol::NamedTuple, newsol::Nothing, pddld, problem, fminibatch)
	verbose && println(@red "new is unsolved: $((oldsol.stats.sol_length, oldsol.stats.expanded))")
	# stats = (;solution_time = oldsol.stats.solution_time,
	# sol_length = oldsol.stats.sol_length,
	# expanded = 10*oldsol.stats.expanded, # let's try to increase priority of this poor guy
	# )
	oldsol
end

issolved(::Nothing) = false
issolved(s::NamedTuple{(:minibatch, :stats)}) = s.stats.solved

function show_stats(solutions)
	solved = filter(issolved, solutions)
	if isempty(solved)
		println(" solved instances: ", length(solved))
		return
	end
	stats = [s.stats for s in solved]
	mean_time = round(mean(s.solution_time for s in stats), digits = 2)
	mean_length = round(mean(s.sol_length for s in stats), digits = 2)
	mean_expanded = round(mean(s.expanded for s in stats), digits = 2)
	println(" solved instances: ", length(solved), 
		" (",round(length(solved) / length(solutions), digits = 2), ") mean length ",mean_length, " mean expanded ",mean_expanded, " mean_time = ", round(mean_time, digits = 3))
end

function _show_stats(stats)
	solved = filter(issolved, stats)
	if isempty(solved)
		println(" solved instances: ", length(solved))
		return
	end
	mean_time = round(mean(s.solution_time for s in solved), digits = 2)
	mean_length = round(mean(s.sol_length for s in solved), digits = 2)
	mean_expanded = round(mean(s.expanded for s in solved), digits = 2)
	mean_excess = round(mean(s.expanded ./ s.sol_length for s in solved), digits = 2)
	println(" solved instances: ", length(solved), 
		" (",round(length(solved) / length(stats), digits = 2), ") mean length ",mean_length, " mean expanded ",mean_expanded, " mean expanded excess ",mean_excess, " mean_time = ", round(mean_time, digits = 3))
end
