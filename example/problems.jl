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

function getproblem(problem)
	problem == "blocks" && return(setup_blocks_slaney())
	problem == "ferry" && return(setup_ferry())
	problem == "gripper" && return(setup_gripper())
	problem == "npuzzle" && return(setup_n_puzzle())
	problem == "zenotravel" && return(setup_zenotravel())
	error("unknown problem $(problem)")
end



