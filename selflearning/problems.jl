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

const WORKING_CLASSIC = Set(["agricola-opt18", "agricola-sat18", "airport-adl", "assembly", "barman-opt11-strips", "barman-opt14-strips", "barman-sat11-strips", "barman-sat14-strips", "blocks", "blocks-3op", "briefcaseworld", "caldera-opt18", "caldera-sat18", "caldera-split-opt18", "caldera-split-sat18", "cavediving", "childsnack-opt14-strips", "childsnack-sat14-strips", "citycar-opt14-adl", "citycar-sat14-adl", "data-network-opt18", "data-network-sat18", "depot", "driverlog", "elevators-00-full", "elevators-00-strips", "elevators-opt11-strips", "elevators-sat11-strips", "ferry", "floortile-opt11-strips", "floortile-opt14-strips", "floortile-sat11-strips", "floortile-sat14-strips", "freecell", "ged-opt14-strips", "ged-sat14-strips", "grid", "gripper", "hanoi", "hiking-opt14-strips", "hiking-sat14-strips", "logistics00", "logistics98", "maintenance-opt14-adl", "maintenance-sat14-adl", "miconic", "miconic-fulladl", "miconic-simpleadl", "movie", "mprime", "mystery", "no-mprime", "no-mystery", "nomystery-opt11-strips", "nomystery-sat11-strips", "nurikabe-opt18", "nurikabe-sat18", "openstacks", "optical-telegraphs", "parking-opt11-strips", "parking-opt14-strips", "parking-sat11-strips", "parking-sat14-strips", "pegsol-opt11-strips", "pegsol-sat11-strips", "philosophers", "pipesworld-06", "pipesworld-notankage", "psr-large", "psr-middle", "rovers", "rovers-02", "satellite", "scanalyzer-opt11-strips", "scanalyzer-sat11-strips", "schedule", "settlers", "settlers-opt18", "settlers-sat18", "snake-opt18", "snake-sat18", "tetris-opt14-strips", "tetris-sat14-strips", "thoughtful-sat14-strips", "tidybot-opt11-strips", "tidybot-opt14-strips", "tidybot-sat11-strips", "tpp", "transport-opt11-strips", "transport-opt14-strips", "transport-sat11-strips", "transport-sat14-strips", "trucks", "tsp", "tyreworld", "visitall-opt11-strips", "visitall-opt14-strips", "visitall-sat11-strips", "visitall-sat14-strips", "woodworking-opt11-strips", "woodworking-sat11-strips", "zenotravel"])
function setup_classic(name)
	benchdir(s...) = joinpath("..","classical-domains","classical", name, s...)
	domain_pddl = benchdir("domain.pddl")
	problem_files  = filter(s -> endswith(s, ".pddl") && s != "domain.pddl", readdir(benchdir()))
	problem_files = [benchdir(s) for s in problem_files]
	ofile(s...) = joinpath("results", name, s...)
	return(domain_pddl, problem_files, ofile)
end

function getproblem(problem)
	problem == "blocks" && return(setup_blocks_slaney())
	problem == "ferry" && return(setup_ferry())
	problem == "gripper" && return(setup_gripper())
	problem == "npuzzle" && return(setup_n_puzzle())
	problem == "zenotravel" && return(setup_zenotravel())
	problem ∈ WORKING_CLASSIC && return(setup_classic(problem))
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

