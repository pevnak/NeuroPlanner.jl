"""
File with utility functions related to working with problems
"""
function save_plan(ofile, plan)
	open(ofile,"w") do fio 
		for a in plan
			s = "("*join([a.name, a.args...]," ")*")"
			println(fio, s)
		end
	end
end

plan_file(s) = endswith(s,".pddl") ? s[1:end-5]*".plan" : s

function load_plan(pfile)
	pfile = plan_file(pfile)
	map(readlines(pfile)) do s
		p = Symbol.(split(s[2:end-1]," "))
		Compound(p[1], Const.(p[2:end]))
	end
end

# for ifile in filter(s -> endswith(s,"plan"), readdir())
# 	if length(unique(readlines(ifile)))>1
# 		@show ifile
# 		continue
# 	end
# 	if length(unique(readlines(ifile)))<1
# 		@show ifile
# 		continue
# 	end
# 	ss = only(unique(readlines(ifile)))
# 	ss = split(ss[2:end-1],",")
# 	plan = map(ss) do s
# 		s = s[findfirst("(",s).stop+1:findfirst(")",s).stop-1]
# 		p = Symbol.(split(s," "))
# 		Compound(p[1], Const.(p[2:end]))
# 	end
# 	save_plan(ifile, plan)
# end


function setup_problem(problem_name)
	sdir(s...) = joinpath(dirname(pathof(NeuroPlanner)),"..", "domains",problem_name, s...)
	domain_pddl = sdir("domain.pddl")
	problem_files = [sdir(f) for f in readdir(sdir()) if endswith(f,".pddl") && f !== "domain.pddl"]
	return(domain_pddl, problem_files)
end

const WORKING_CLASSIC = Set(["agricola-opt18", "agricola-sat18", "airport-adl", "assembly", "barman-opt11-strips", "barman-opt14-strips", "barman-sat11-strips", "barman-sat14-strips", "blocks", "blocks-3op", "briefcaseworld", "caldera-opt18", "caldera-sat18", "caldera-split-opt18", "caldera-split-sat18", "cavediving", "childsnack-opt14-strips", "childsnack-sat14-strips", "citycar-opt14-adl", "citycar-sat14-adl", "data-network-opt18", "data-network-sat18", "depot", "driverlog", "elevators-00-full", "elevators-00-strips", "elevators-opt11-strips", "elevators-sat11-strips", "ferry", "floortile-opt11-strips", "floortile-opt14-strips", "floortile-sat11-strips", "floortile-sat14-strips", "freecell", "ged-opt14-strips", "ged-sat14-strips", "grid", "gripper", "hanoi", "hiking-opt14-strips", "hiking-sat14-strips", "logistics00", "logistics98", "maintenance-opt14-adl", "maintenance-sat14-adl", "miconic", "miconic-fulladl", "miconic-simpleadl", "movie", "mprime", "mystery", "no-mprime", "no-mystery", "nomystery-opt11-strips", "nomystery-sat11-strips", "nurikabe-opt18", "nurikabe-sat18", "openstacks", "optical-telegraphs", "parking-opt11-strips", "parking-opt14-strips", "parking-sat11-strips", "parking-sat14-strips", "pegsol-opt11-strips", "pegsol-sat11-strips", "philosophers", "pipesworld-06", "pipesworld-notankage", "psr-large", "psr-middle", "rovers", "rovers-02", "satellite", "scanalyzer-opt11-strips", "scanalyzer-sat11-strips", "schedule", "settlers", "settlers-opt18", "settlers-sat18", "snake-opt18", "snake-sat18", "tetris-opt14-strips", "tetris-sat14-strips", "thoughtful-sat14-strips", "tidybot-opt11-strips", "tidybot-opt14-strips", "tidybot-sat11-strips", "tpp", "transport-opt11-strips", "transport-opt14-strips", "transport-sat11-strips", "transport-sat14-strips", "trucks", "tsp", "tyreworld", "visitall-opt11-strips", "visitall-opt14-strips", "visitall-sat11-strips", "visitall-sat14-strips", "woodworking-opt11-strips", "woodworking-sat11-strips", "zenotravel"])
const IPC_PROBLEM_NAMES = ["ferry", "rovers","blocksworld","floortile","satellite","spanner","childsnack","miconic","sokoban","transport"]
const IPC_PROBLEMS = ["ipc23_"*s for s in IPC_PROBLEM_NAMES]

function setup_classic(name)
	benchdir(s...) = joinpath("..","classical-domains","classical", name, s...)
	domain_pddl = benchdir("domain.pddl")
	problem_files  = filter(s -> endswith(s, ".pddl") && s != "domain.pddl", readdir(benchdir()))
	problem_files = [benchdir(s) for s in problem_files]
	return(domain_pddl, problem_files)
end


function setup_ispc23_problem(problem_name)
	sdir(s...) = joinpath("..","ipc23-learning",problem_name,s...)
	domain_pddl = sdir("domain.pddl")
	problem_files = mapreduce(vcat, [joinpath("testing","easy"), joinpath("testing","medium"), joinpath("testing","hard"), joinpath("training","easy")]) do s  
		[sdir(s, f) for f in readdir(sdir(s)) if endswith(f,".pddl") && f !== "domain.pddl"]
	end
	return(domain_pddl, problem_files)
end

function getproblem(problem)
	problem == "blocks" && return(setup_problem("blocks-slaney"))
	problem == "ferry" && return(setup_problem("ferry"))
	problem == "gripper" && return(setup_problem("gripper"))
	problem == "npuzzle" && return(setup_problem("n-puzzle"))
	problem == "zenotravel" && return(setup_problem("zenotravel"))
	problem == "spanner" && return(setup_problem("spanner"))
	problem == "elevators_00" && return(setup_problem("elevators-00-strips"))
	problem == "elevators_11" && return(setup_problem("elevators-opt11-strips"))
	problem ∈ IPC_PROBLEMS && return(setup_ispc23_problem(problem[7:end]))
	# problem ∈ WORKING_CLASSIC && return(setup_problem("blocks-slaney"))
	error("unknown problem $(problem)")
end

"""
(domain_pddl, problem_files) =  getproblem(problem)
(domain_pddl, problem_files) =  getproblem(problem, sort_by_complexity)

if sort_by_complexity is true, problem_files are sorted by the number of objects, 
which is treated as a proxy for complexity.
"""
function getproblem(problem, sort_by_complexity)
	!sort_by_complexity	 && return(getproblem(problem))
	(domain_pddl, problem_files) = getproblem(problem)
	no = map(f -> length(load_problem(f).objects), problem_files)
	problem_files = problem_files[sortperm(no)]
	return(domain_pddl, problem_files)
end


"""
_serialize_plans(problem_name)

works only for Leah's naming convention
"""
function accomodate_leah_plans(problem_name)
	benchdir(s...) = joinpath("benchmarks", problem_name, s...)
	domain_pddl = benchdir("domain.pddl")
	problem_files = filter(s -> endswith(s, ".pddl") && (s != "domain.pddl"), readdir(benchdir()))
	sdir(s...) = joinpath("..","domains",problem_name, s...)
	!isdir(sdir()) && mkpath(sdir())
	run(`cp $(domain_pddl) $(sdir("domain.pddl"))`)
	for ifile in problem_files
		ofile = sdir(ifile)
		run(`cp $(benchdir(ifile)) $(ofile)`)
		pfile = benchdir("sol"*ifile[5:end-5])
		if isfile(pfile)
			run(`cp $(pfile) $(sdir(plan_file(ifile)))`)
		end
	end
end

function merge_ferber_problems(problem_name)
	benchdir(s...) = joinpath("benchmarks", problem_name, s...)
	domain_pddl = benchdir("domain.pddl")
	problem_files = filter(s -> endswith(s, ".pddl") && (s != "domain.pddl"), readdir(benchdir()))
	sdir(s...) = joinpath("..","domains",problem_name, s...)
	!isdir(sdir()) && mkpath(sdir())
	run(`cp $(domain_pddl) $(sdir("domain.pddl"))`)
	for ifile in problem_files
		ofile = sdir(ifile)
		run(`cp $(benchdir(ifile)) $(ofile)`)
		pfile = benchdir("sol"*ifile[5:end-5])
		if isfile(pfile)
			run(`cp $(pfile) $(sdir(plan_file(ifile)))`)
		end
	end
end

hashfile(f) = open(hash ∘ read, f, "r")

function merge_ferber_problems()
	domains = filter(s -> isdir("../ferber/"*s), readdir("../ferber/"))
	for domain in domains
		sub_problems = filter(s -> isdir(joinpath("../ferber", domain, s)), readdir("../ferber/"*domain))
		domain_files = unique(hashfile, joinpath("../ferber", domain, sub_problem,"domain.pddl") for sub_problem in sub_problems); 
		length(domain_files) == 1 || println("domain.pddl files in $(domain)  are different")
		ispath("../domains/$(domain)") && error("$(domain) exists")
		mkpath("../domains/$(domain)")
		run(`cp $(joinpath("../ferber", domain, first(sub_problems),"domain.pddl")) ../domains/$(domain)/domain.pddl`)
		for sub_problem in sub_problems
			for s in readdir(joinpath("../ferber", domain, sub_problem))
				match(r"^p[0-9]+.pddl", s) === nothing && continue
				run(`cp $(joinpath("../ferber", domain, sub_problem,s)) ../domains/$(domain)/$(sub_problem)_$(s)`)
			end
		end
	end
end


function _parse_plan(domain, problem_file, plan_file)
	problem = load_problem(problem_file)
	state = initstate(domain, problem)
	goal = goalstate(domain, problem)
	plan = map(readlines(plan_file)) do s
		acts = available(domain, state)
		p = Symbol.(split(s[2:end-1]," "))
		a = Compound(p[1], Const.(p[2:end]))
		a ∉ acts && error("action from plan not in available actions")
		state = execute(domain, state, a)
		a
	end
	@assert issubset(goal, state) "goal was not reached"
	plan
end

function systematize(problem)
	domain_pddl, problem_files = getproblem(problem)
	problem_name = split(domain_pddl,"/")[2]
	sdir(s...) = joinpath("..","domains",problem_name, s...)
	!isdir(sdir()) && mkpath(sdir())
	run(`cp $(domain_pddl) $(sdir("domain.pddl"))`)
	for ifile in problem_files
		s = split(ifile, "/")
		ofile = sdir(join(s[3:end],"_"))
		run(`cp $(ifile) $(ofile)`)
		if isfile(plan_file(problem_name, ifile))
			plan = deserialize(plan_file(problem_name, ifile)).plan
			save_plan(ofile[1:end-5]*".plan", plan)
		end
	end
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


