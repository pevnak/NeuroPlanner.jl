using Serialization
using Statistics
using DataFrames
using NeuroPlanner
using JSON
using CSV
using DataFrames
using HypothesisTests
using Statistics
using PrettyTables

function lexargmax(a, b)
	@assert length(a) == length(b)
	function _lt(i,j)
		a[i] == a[j] && return(b[i] < b[j])
		a[i] < a[j]
	end
	first(sort(1:length(a), lt = _lt, rev = true))
end

function show_stats(combined_stats, key)
	key ∉ (:tst_solved, :expanded, :solution_time, :sol_length) && error("works only for tst_solved and :expanded are supported")
	k1 = Symbol("$(key)_fold1")
	k2 = Symbol("$(key)_fold2")
	dfs = mapreduce((args...) -> outerjoin(args..., on = :domain_name), unique(combined_stats.loss_name)) do ln 
		df1 = filter(r -> r.loss_name == ln, combined_stats)
		dfs = mapreduce((args...) -> outerjoin(args..., on = :domain_name), unique(df1.arch_name)) do an 
			df2 = filter(r -> r.arch_name == an, df1)
			gdf = DataFrames.groupby(df2, :domain_name)
			a = combine(gdf) do sub_df
				# (;average = mean(sub_df.tst_solved), maximum = maximum(sub_df.tst_solved))
				v = map(unique(sub_df.seed)) do s
					subsub_df = filter(r -> r.seed == s, sub_df)

					# i = subsub_df[argmax(subsub_df.tst_solved_fold1),k2]
					# j = subsub_df[argmax(subsub_df.tst_solved_fold2),k1]
					i = subsub_df[lexargmax(subsub_df.tst_solved_fold1, -subsub_df.sol_length_fold1),k2]
					j = subsub_df[lexargmax(subsub_df.tst_solved_fold2, -subsub_df.sol_length_fold2),k1]
					(i+j) / 2
				end |> mean
				DataFrame("$ln $an" => [v])
			end
		end
		dfs
	end;
end

###########
#	Collect all stats to one big DataFrame, over which we will perform queries
###########

domain_name = "gripper"

# function read_data(domain_name)
# 	suffix = "_stats.jls"
# 	files = readdir(joinpath("super",domain_name))
# 	files = filter(s -> endswith(s, suffix), files)
# 	map(files) do f 
# 		prefix = f[1:end-length(suffix)]
# 		df = DataFrame(vec(deserialize(joinpath("super",domain_name,f))))
# 		parameters = deserialize(joinpath("super",domain_name,prefix*"_settings.jls"))
# 	end
# end



loss_name = "lstar"
arch_name = "pdd"
max_time = 30
graph_layers = 1
dense_dim = 32
dense_layers = 2
residual = "none"
seed = 1

function submit_missing(;domain_name, arch_name, loss_name, max_steps,  max_time, graph_layers, residual, dense_layers, dense_dim, seed)
	filename = joinpath("super", domain_name, join([arch_name, loss_name, max_steps,  max_time, graph_layers, residual, dense_layers, dense_dim, seed], "_"))
	filename = filename*"_stats.jls"
	isfile(filename) && return(false)
	run(`sbatch -D /home/pevnytom/julia/Pkg/NeuroPlanner.jl/selflearning slurm.sh $domain_name $dense_dim $graph_layers $residual $loss_name $arch_name`)
	return(true)
end

function read_data(;domain_name, arch_name, loss_name, max_steps,  max_time, graph_layers, residual, dense_layers, dense_dim, seed)
	# filename = joinpath("super", domain_name, join([arch_name, loss_name, max_steps,  max_time, graph_layers, residual, dense_layers, dense_dim, seed], "_"))
	filename = joinpath("super", domain_name, join([arch_name, loss_name, max_steps,  max_time, graph_layers, residual, dense_layers, dense_dim, seed], "_"))
	filename = filename*"_stats.jls"
	!isfile(filename) && return(DataFrame())
	df = DataFrame(vec(deserialize(filename)))
	select!(df, Not(:trajectory))
	df[:,:domain_name] .= domain_name
	df[:,:arch_name] .= arch_name
	df[:,:loss_name] .= loss_name
	df[:,:graph_layers] .= graph_layers
	df[:,:dense_dim] .= dense_dim
	df[:,:dense_layers] .= dense_layers
	df[:,:residual] .= residual
	df[:,:seed] .= seed
	df[:,:max_steps] .= max_steps
	df[:,:max_time] .= max_time
	df
end

function update_data(;domain_name, arch_name, loss_name, max_steps,  max_time, graph_layers, residual, dense_layers, dense_dim, seed)
	filename = joinpath("super", domain_name, join([arch_name, loss_name, max_steps,  max_time, graph_layers, residual, dense_layers, dense_dim, seed], "_"))
	filename = filename*"_stats.jls"
	!isfile(filename) && return(DataFrame())
	d = dirname(getproblem(domain_name)[1])
	domain = load_domain(getproblem(domain_name)[1])
	df = vec(deserialize(filename))
	for row in df 
		row.trajectory === nothing && continue
		problem_file = joinpath(d, basename(row.problem_file))
		pfile = problem_file[1:end-5]*".plan"
		if !isfile(pfile) || (length(readlines(pfile)) +1 < length(row.trajectory))
			println("updating ", pfile)
			problem = load_problem(problem_file)
			plan = NeuroPlanner.plan_from_trajectory(domain, problem, row.trajectory)
			save_plan(pfile, plan)
		end
	end
end

max_steps = 10_000
max_time = 30
dense_layers = 2
seed = 1
problems = ["blocks","ferry","npuzzle","gripper", "spanner","elevators_00","elevators_11"]
# stats = map(Iterators.product(("asnet","pddl", "hgnnlite", "hgnn"), ("lstar","l2","lrt","lgbfs"), problems, (4, 8, 16), (1, 2, 3), (:none, :linear), (1, 2, 3))) do (arch_name, loss_name, domain_name, dense_dim, graph_layers, residual, seed)
stats = map(Iterators.product(("levinasnet",), ("levinloss",), problems, (4, 8, 16), (1, 2, 3), (:none, :linear, :dense), (1, 2, 3))) do (arch_name, loss_name, domain_name, dense_dim, graph_layers, residual, seed)
	# submit_missing(;domain_name, arch_name, loss_name, max_steps,  max_time, graph_layers, residual, dense_layers, dense_dim, seed)
	read_data(;domain_name, arch_name, loss_name, max_steps,  max_time, graph_layers, residual, dense_layers, dense_dim, seed)
end;
df = reduce(vcat, filter(!isempty, vec(stats)))

stats = map(Iterators.product(("hgnn",), ("bellman",), problems, (4, 8, 16), (1, 2, 3), (:none, :linear, :dense), (1, 2, 3))) do (arch_name, loss_name, domain_name, dense_dim, graph_layers, residual, seed)
	# submit_missing(;domain_name, arch_name, loss_name, max_steps,  max_time, graph_layers, residual, dense_layers, dense_dim, seed)
	read_data(;domain_name, arch_name, loss_name, max_steps,  max_time, graph_layers, residual, dense_layers, dense_dim, seed)
end;
df = reduce(vcat, filter(!isempty, vec(stats)))

gdf = DataFrames.groupby(df, [:domain_name, :arch_name, :loss_name, :planner, :max_steps,  :max_time, :graph_layers, :residual, :dense_layers, :dense_dim, :seed])
combined_stats = combine(gdf) do sub_df
	ii = sortperm(sub_df.problem_file)
	i1 = filter(i -> !sub_df.used_in_train[i], ii[1:2:end])
	i2 = filter(i -> !sub_df.used_in_train[i], ii[2:2:end])
	(;	trn_solved = mean(sub_df.solved[sub_df.used_in_train]),
     	tst_solved = mean(sub_df.solved[.!sub_df.used_in_train]),
     	tst_solved_fold1 = mean(sub_df.solved[i1]),
     	tst_solved_fold2 = mean(sub_df.solved[i2]),
     	expanded_fold1 = mean(sub_df.expanded[i1]),
     	expanded_fold2 = mean(sub_df.expanded[i2]),
     	solution_time_fold1 = mean(sub_df.solution_time[i1]),
     	solution_time_fold2 = mean(sub_df.solution_time[i2]),
     	sol_length_fold1 = mean(skipmissing(sub_df.sol_length[i1])),
     	sol_length_fold2 = mean(skipmissing(sub_df.sol_length[i2])),
     	time_in_heuristic = mean(sub_df.time_in_heuristic[.!sub_df.used_in_train]),
     	solved_problems = size(sub_df, 1),
	)
end

dd = show_stats(filter(r -> r.planner == "AStarPlanner", combined_stats), :tst_solved)

# dd = show_stats(filter(r -> r.planner == "GreedyPlanner", combined_stats), :tst_solved)
# dd = show_stats(filter(r -> r.planner == "AStarPlanner", combined_stats), :tst_solved)
ii = filter(s -> endswith(s,"hgnn"), names(dd));
dd[:,["domain_name",ii...]]
ii = filter(s -> endswith(s,"asnet"), names(dd));
dd[:,["domain_name",ii...]]
ii = filter(s -> endswith(s,"pddl"), names(dd));
dd[:,["domain_name",ii...]]
