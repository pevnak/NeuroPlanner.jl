using PDDL
using SymbolicPlanners
using Serialization
using Random

include("problems.jl")

function plan_file(problem_file)
	middle_path = splitpath(problem_file)[3:end-1]
	middle_path = filter(∉(["problems"]),middle_path)
	joinpath("plans", problem_name, middle_path..., basename(problem_file)[1:end-5]*".jls")
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

# problem_name = "gripper"
problem_name = "blocks"
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
