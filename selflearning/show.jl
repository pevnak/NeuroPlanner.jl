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
using Printf

function lexargmax(a, b)
	@assert length(a) == length(b)
	function _lt(i,j)
		a[i] == a[j] && return(b[i] < b[j])
		a[i] < a[j]
	end
	first(sort(1:length(a), lt = _lt, rev = true))
end

function compute_stats(df; max_time = 30)
	gdf = DataFrames.groupby(df, [:domain_name, :arch_name, :loss_name, :planner, :max_steps,  :max_time, :graph_layers, :residual, :dense_layers, :dense_dim, :seed])
	combined_stats = combine(gdf) do sub_df
		ii = sortperm(sub_df.problem_file)
		i1 = filter(i -> !sub_df.used_in_train[i], ii[1:2:end])
		i2 = filter(i -> !sub_df.used_in_train[i], ii[2:2:end])
		solved = sub_df.solution_time .≤ max_time
		(;	trn_solved = mean(solved[sub_df.used_in_train]),
	     	tst_solved = mean(solved[.!sub_df.used_in_train]),
	     	tst_solved_fold1 = mean(solved[i1]),
	     	tst_solved_fold2 = mean(solved[i2]),
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
end

function show_loss(combined_stats, key; percents = false)
	key ∉ (:tst_solved, :expanded, :solution_time, :sol_length) && error("works only for tst_solved and :expanded are supported")
	k1 = Symbol("$(key)_fold1")
	k2 = Symbol("$(key)_fold2")
	dfs = mapreduce((args...) -> outerjoin(args..., on = :domain_name), unique(combined_stats.loss_name)) do ln 
		df1 = filter(r -> r.loss_name == ln, combined_stats)
		gdf = DataFrames.groupby(df1, :domain_name)
		dfs = combine(gdf) do sub_df
			# (;average = mean(sub_df.tst_solved), maximum = maximum(sub_df.tst_solved))
			v = map(unique(sub_df.seed)) do s
				subsub_df = filter(r -> r.seed == s, sub_df)
				i = subsub_df[lexargmax(subsub_df.tst_solved_fold1, -subsub_df.sol_length_fold1),k2]
				j = subsub_df[lexargmax(subsub_df.tst_solved_fold2, -subsub_df.sol_length_fold2),k1]
				(i+j) / 2
			end |> mean
			# v = percents ? round(100*v, digits = 2) : v
			v = percents ? round(Int, 100*v) : round(v, digits = 2)
			DataFrame("$ln" => [v])
		end
		dfs
	end;
end

function high_max(data, i, j)
	j == 1 && return(false)
	# i == size(data, 1)  ? minimum(data[i,2:end]) == data[i,j] : maximum(data[i,2:end]) == data[i,j]
	maximum(data[i,2:end]) == data[i,j]
end

function high_min(data, i, j)
	j == 1 && return(false)
	# i == size(data, 1)  ? minimum(data[i,2:end]) == data[i,j] : maximum(data[i,2:end]) == data[i,j]
	minimum(data[i,2:end]) == data[i,j]
end

###########
#	Collect all stats to one big DataFrame, over which we will perform queries
###########
loss_name = "lstar"
arch_name = "pddl"
max_time = 30
graph_layers = 1
dense_dim = 32
dense_layers = 2
residual = "none"
seed = 1

function submit_missing(;dry_run = true, domain_name, arch_name, loss_name, max_steps,  max_time, graph_layers, residual, dense_layers, dense_dim, seed)
	filename = joinpath("super", domain_name, join([arch_name, loss_name, max_steps,  max_time, graph_layers, residual, dense_layers, dense_dim, seed], "_"))
	filename = filename*"_stats.jls"
	isfile(filename) && return(false)
	@show filename
	dry_run || run(`sbatch -D /home/pevnytom/julia/Pkg/NeuroPlanner.jl/selflearning slurm.sh $domain_name $dense_dim $graph_layers $residual $loss_name $arch_name $seed`)
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

function add_rank(df)
	r = map(1:size(df,1)) do i 
		StatsBase.competerank([df[i,j] for j in 2:size(df,2)]; rev = true)
	end
	push!(df, ["rank ", mean(r)...])
end

max_steps = 10_000
max_time = 30
dense_layers = 2
seed = 1
dry_run = true
problems = ["blocks","ferry","npuzzle","spanner","elevators_00"]
stats = map(Iterators.product(("hgnn",), ("bellman", "lstar", "l2", "lrt", "lgbfs",), problems, (4, 8, 16), (1, 2, 3), (:none, :linear), (1, 2, 3))) do (arch_name, loss_name, domain_name, dense_dim, graph_layers, residual, seed)
	# submit_missing(;dry_run, domain_name, arch_name, loss_name, max_steps,  max_time, graph_layers, residual, dense_layers, dense_dim, seed)
	read_data(;domain_name, arch_name, loss_name, max_steps,  max_time, graph_layers, residual, dense_layers, dense_dim, seed)
end |> vec

lstats = map(Iterators.product(("levinasnet",), ("levinloss",), problems, (4, 8, 16), (1, 2, 3), (:none, :linear, :dense), (1, 2, 3))) do (arch_name, loss_name, domain_name, dense_dim, graph_layers, residual, seed)
	# submit_missing(;dry_run, domain_name, arch_name, loss_name, max_steps,  max_time, graph_layers, residual, dense_layers, dense_dim, seed)
	read_data(;domain_name, arch_name, loss_name, max_steps,  max_time, graph_layers, residual, dense_layers, dense_dim, seed)
end |> vec;

df = reduce(vcat, filter(!isempty, vec(vcat(stats, lstats))))

function make_table(df, k, percents, high; max_time = 30, backend = Val(:text))
	ddf = filter(r -> r.planner == "AStarPlanner", df);
	da = show_loss(compute_stats(ddf;max_time), k; percents);
	da = da[:,[:domain_name, :lstar, :lgbfs, :lrt, :l2, :bellman]]
	# da = add_rank(show_loss(compute_stats(ddf), k; percents = true));

	ddf = filter(r -> r.planner == "GreedyPlanner" && r.arch_name == "hgnn", df)
	d1 = show_loss(compute_stats(ddf;max_time), k; percents)
	ddf = filter(r -> r.arch_name == "levinasnet", df)
	d2 = show_loss(compute_stats(ddf;max_time), k; percents)
	ddf = outerjoin(d1, d2, on = :domain_name)
	db = ddf[:,[:domain_name, :lstar, :lgbfs, :lrt, :l2, :bellman, :levinloss]]
	# db = add_rank(ddf)


	header = [names(da)..., " ", names(db)[2:end]...]
	data = hcat(Matrix(da), fill(" ", size(da,1)), Matrix(db)[:,2:end])
	ha = [high(Matrix(da), i, j) for i in 1:size(da,1), j in 1:size(da,2)]
	hb = [high(Matrix(db), i, j) for i in 1:size(db,1), j in 1:size(db,2)]
	h = hcat(ha, falses(size(da,1)), hb[:,2:end])

	highlighters = backend == Val(:latex) ? LatexHighlighter((data, i, j) -> h[i, j], "textbf") :  Highlighter((data, i, j) -> h[i, j], crayon"yellow")
	pretty_table(data ; header, backend , highlighters)
end

df = CSV.read("super/results.csv", DataFrame)

ddf = filter(r -> r.arch_name ∈ ("hgnn", "levinasnet"), df);
make_table(ddf, :tst_solved, true, high_max; max_time =  30)

make_table(ddf, :solution_time, false, high_min)
make_table(ddf, :expanded, false, high_min)
make_table(ddf, :sol_length, false, high_min)

make_table(ddf, :tst_solved, true, high_max; backend = Val(:latex), max_time =  5)
