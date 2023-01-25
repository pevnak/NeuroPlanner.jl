using PDDL
using SymbolicPlanners
using Serialization
using Random
using NeuroPlanner: plan_from_trajectory

include("problems.jl")

function plan_file(problem_file)
	middle_path = splitpath(problem_file)[3:end-1]
	middle_path = filter(∉(["problems"]),middle_path)
	joinpath("plans", problem_name, middle_path..., basename(problem_file)[1:end-5]*".jls")
end

"""
	add solutions from solved problems
"""
function harmonize_trajectories(df, problem_name)
	for (i, row) in enumerate(eachrow(df))
		!row.solved && continue
		trajectory = row.trajectory
		pl_file = "plans/$(problem_name)/$(basename(row.problem_file)[1:end-5]*".jls")"
		problem = load_problem(row.problem_file)
		new_plan = plan_from_trajectory(domain, problem, trajectory)
		#get plan from the trajectory
		if !isfile(pl_file)
			serialize(pl_file, (;plan = new_plan))
		else 
			old_plan = deserialize(pl_file).plan
			if !verify_plan(domain, problem, old_plan)
				rm(pl_file)
				serialize(pl_file, (;plan = new_plan))
			else 
				if length(new_plan) < length(old_plan)
					serialize(pl_file, (;plan = new_plan))
				end
			end
		end
	end
end

function solveproblem(domain, problem_file)
	problem = load_problem(problem_file)

	state = initstate(domain, problem)
	spec = MinStepsGoal(problem)
	planner = AStarPlanner(HAdd(); max_time=3600, save_search = true)
	# planner = AStarPlanner(HMax(); max_time=3600, save_search = true)

	goal = PDDL.get_goal(problem)
	t = @elapsed sol = planner(domain, state, spec)
	(sol, t)
end

problem_name = "gripper"
# problem_name = "blocks"
domain_pddl, problem_files, _ = getproblem(problem_name, false)
domain = load_domain(domain_pddl)
sol, t = solveproblem(domain, first(problem_files))

problem_files = filter(s -> !isfile(plan_file(s)), problem_files)
problem_files = shuffle(problem_files)

for problem_file in problem_files
	f = plan_file(problem_file)
	sol, t = solveproblem(domain, problem_file)
	println("solving ", basename(problem_file), " took ", t,"s")
	sol.status != :success && continue
	plan = sol.plan
	expanded = sol.expanded
	!isdir(dirname(f)) && mkpath(dirname(f))
	serialize(f, (;plan, t, expanded))
end



for d in readdir("plans/blocks")
	for f in readdir("plans/blocks/$(d)")
		endswith(f,".jls") && continue
		run(`mv plans/blocks/$(d)/$(f) plans/blocks/$(d)/$(f).jls`)
	end
end
