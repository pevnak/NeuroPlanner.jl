using Printf
function setup_blocks_slaney()
	benchdir(s...) = joinpath("benchmarks","blocks-slaney",s...)
	taskfiles(s) = [benchdir(s, f) for f in readdir(benchdir(s)) if startswith(f,"task",)]
	domain_pddl = benchdir("domain.pddl")
	problem_files = mapreduce(taskfiles, vcat, ["blocks$(i)" for i in 3:15])
	ofile(s...) = joinpath("results", "blocks-slaney", s...)
	return(domain_pddl, problem_files, ofile)
end

function setup_ferry()
	benchdir(s...) = joinpath("benchmarks","ferry",s...)
	domain_pddl = benchdir("ferry.pddl")
	problem_files = [benchdir(s,f) for s in ["train","test"] for f in readdir(benchdir(s)) ]
	problem_files = problem_files[sortperm([parse(Int,split(s,"-")[2][2:end]) for s in problem_files])]
	ofile(s...) = joinpath("results", "ferry", s...)
	return(domain_pddl, problem_files, ofile)
end

function setup_gripper()
	benchdir(s...) = joinpath("benchmarks","gripper",s...)
	domain_pddl = benchdir("domain.pddl")
	problem_files = [benchdir("problems", f) for f in readdir(benchdir("problems"))]
	ofile(s...) = joinpath("results", "gripper", s...)
	return(domain_pddl, problem_files, ofile)
end

function setup_n_puzzle()
	benchdir(s...) = joinpath("benchmarks","n-puzzle",s...)
	domain_pddl = benchdir("domain.pddl")
	problem_files = [benchdir(s,f) for s in ["train","test"] for f in readdir(benchdir(s)) ]
	problem_files = problem_files[sortperm([parse(Int,split(s,"-")[4][1:1]) for s in problem_files])]
	ofile(s...) = joinpath("results", "n-puzzle", s...)
	return(domain_pddl, problem_files, ofile)
end

function setup_zenotravel()
	benchdir(s...) = joinpath("benchmarks","zenotravel",s...)
	domain_pddl = benchdir("domain.pddl")
	problem_files = [benchdir(s,f) for s in ["train","test"] for f in readdir(benchdir(s)) ]
	problem_files = problem_files[sortperm([parse(Int,split(s,"-")[2][end:end]) for s in problem_files])]
	ofile(s...) = joinpath("results", "zenotravel", s...)
	return(domain_pddl, problem_files, ofile)
end

function setup_ispc(name)
	benchdir(s...) = joinpath("..","classical-domains","classical", name, s...)
	domain_pddl = benchdir("domain.pddl")
	problem_files = [benchdir(@sprintf("p%02d.pddl", 1)) for i in 1:20]
	ofile(s...) = joinpath("results", name, s...)
	return(domain_pddl, problem_files, ofile)
end

function getproblem(problem)
	problem == "blocks" && return(setup_blocks_slaney())
	problem == "ferry" && return(setup_ferry())
	problem == "gripper" && return(setup_gripper())
	problem == "npuzzle" && return(setup_n_puzzle())
	problem == "zenotravel" && return(setup_zenotravel())
	standard_ispc = Set(["agricola-sat18","caldera-sat18","woodworking-sat11-strips"])
	problem ∈ standard_ispc && return(setup_ispc(problem))
	error("unknown problem $(problem)")
end

"""
(domain_pddl, problem_files, ofile) =  getproblem(problem)
(domain_pddl, problem_files, ofile) =  getproblem(problem, sort_by_complexity)

if sort_by_complexity is true, problem_files are sorted by the number of objects, 
which is treated as a proxy for complexity.
"""
function getproblem(problem, sort_by_complexity)
	!sort_by_complexity	 && return(getproblem(problem))
	(domain_pddl, problem_files, ofile) = getproblem(problem)
	no = map(f -> length(load_problem(f).objects), problem_files)
	problem_files = problem_files[sortperm(no)]
	return(domain_pddl, problem_files, ofile)
end

"""
	A leftover function, which is used to found out, how many domains have similar 
	relations and similar names of objects.
"""
function similarity_of_problems()
	prefix = "../classical-domains/classical"
	problems =filter(isdir, readdir(prefix, join = true))
	problems = filter(f -> isfile(joinpath(f, "domain.pddl")), problems)

	constants = []
	nunanary_predicates = []
	binary_predicates = []
	for f in problems
		try 
			domain = load_domain(joinpath(f, "domain.pddl"))
		catch 
			@info "cannot parse $(f)"
		end
		predicates = collect(domain.predicates)
		append!(binary_predicates, [kv[2] for kv in predicates if length(kv[2].args) == 2])
		append!(nunanary_predicates, [kv[2] for kv in predicates if length(kv[2].args) ≤ 2])
		append!(constants, domain.constants)
	end
end


"""
plan_file(problem_file)

a path to a plan of 
"""
function plan_file(problem_name, problem_file)
	middle_path = splitpath(problem_file)[3:end-1]
	middle_path = filter(∉(["problems"]),middle_path)
	middle_path = filter(∉(["test"]),middle_path)
	joinpath("plans", problem_name, middle_path..., basename(problem_file)[1:end-5]*".jls")
end

