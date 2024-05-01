using Serialization
using Statistics
using DataFrames
using NeuroPlanner
using JSON
using CSV
using DataFrames
using Random
using HypothesisTests
using Statistics
using PrettyTables
using Term
using Printf
using StatsBase
IPC_PROBLEM_NAMES = ["ferry", "rovers","blocksworld","floortile","satellite","spanner","childsnack","miconic","sokoban","transport"]
IPC_PROBLEMS = ["ipc23_"*s for s in IPC_PROBLEM_NAMES]

highlight_max(data, i, j) = j > 1 && maximum(skipmissing(data[i,2:end])) == data[i,j]
highlight_min(data, i, j) = j > 1 && minimum(skipmissing(data[i,2:end])) == data[i,j]
nohighlight(args...) = false

function highlight_table(df; backend = Val(:text), high = highlight_max)
	header = [names(df)...]
	data = Matrix(df)
	h = [highlight_max(Matrix(df), i, j) for i in 1:size(df,1), j in 1:size(df,2)]
	h[h .=== missing] .= false
	highlighters = Highlighter((data, i, j) -> h[i, j], crayon"yellow")
	pretty_table(data ; header, backend = Val(:text) , highlighters)
end


joindomains(a,b) = outerjoin(a, b, on = :domain_name)

function compute_stats(df; max_time = 30)
	gdf = DataFrames.groupby(df, [:domain_name, :arch_name, :loss_name, :planner, :max_steps,  :max_time, :graph_layers, :residual, :dense_layers, :dense_dim, :seed])
	combined_stats = combine(gdf) do sub_df
		ii = sortperm(sub_df.problem_file)
		i1 = filter(i -> !sub_df.used_in_train[i], ii[1:2:end])
		i2 = filter(i -> !sub_df.used_in_train[i], ii[2:2:end])
		solved = sub_df.solution_time .≤ max_time
		(;	trn_solved = mean(solved[sub_df.used_in_train]),
			trn_solved_sum = sum(solved[sub_df.used_in_train]),
	     	tst_solved = mean(solved[.!sub_df.used_in_train]),
	     	tst_solved_sum = sum(solved[.!sub_df.used_in_train]),
	     	tst_solved_fold1 = mean(solved[i1]),
	     	tst_solved_fold2 = mean(solved[i2]),
	     	expanded_fold1 = mean(sub_df.expanded[i1]),
	     	expanded_fold2 = mean(sub_df.expanded[i2]),
	     	solution_time_fold1 = mean(sub_df.solution_time[i1]),
	     	solution_time_fold2 = mean(sub_df.solution_time[i2]),
	     	sol_length_fold1 = mean(skipmissing(sub_df.sol_length[i1])),
	     	sol_length_fold2 = mean(skipmissing(sub_df.sol_length[i2])),
	     	solx_length_fold1 = Dict([sub_df.problem_file[i] => sub_df.sol_length[i] for i in i1 if solved[i]]),
	     	solx_length_fold2 = Dict([sub_df.problem_file[i] => sub_df.sol_length[i] for i in i2 if solved[i]]),
	     	time_in_heuristic = mean(sub_df.time_in_heuristic[.!sub_df.used_in_train]),
	     	solved_problems = size(sub_df, 1),
		)
	end
end

function submit_missing(;dry_run = true, domain_name, arch_name, loss_name, max_steps,  max_time, graph_layers, residual, dense_layers, dense_dim, seed, result_dir = "super")
	filename = joinpath(result_dir, domain_name, join([arch_name, loss_name, max_steps,  max_time, graph_layers, residual, dense_layers, dense_dim, seed], "_"))
	if isfile(filename*"_stats.jls")
		println(@green "finished stats "*filename)
	 	return(false)
	end
	if isfile(filename*"_model.jls") && isfile(filename*"_stats_tmp.jls")
		println(@cyan "finished model with temporary stats "*filename)
	 	return(false)
	end

	if isfile(filename*"_model.jls")
		println(@yellow "finished model "*filename)
	 	return(false)
	end
	println(@red "submit "*filename)
	dry_run || run(`sbatch -D /home/pevnytom/julia/Pkg/NeuroPlanner.jl/selflearning slurm.sh $domain_name $dense_dim $graph_layers $residual $loss_name $arch_name $seed`)
	return(true)
end

function read_data(;domain_name, arch_name, loss_name, max_steps,  max_time, graph_layers, residual, dense_layers, dense_dim, seed, result_dir="super")
	# filename = joinpath("super", domain_name, join([arch_name, loss_name, max_steps,  max_time, graph_layers, residual, dense_layers, dense_dim, seed], "_"))
	filename = joinpath(result_dir, domain_name, join([arch_name, loss_name, max_steps,  max_time, graph_layers, residual, dense_layers, dense_dim, seed], "_"))
	filename = filename*"_stats.jls"
	!isfile(filename) && return(DataFrame())
	stats = deserialize(filename)
	df = stats isa DataFrame ? stats : DataFrame(vec(stats))
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

function vitek_table(df, k, high; max_time = 30, backend = Val(:text))
	# df = filter(r -> r.planner == "AStarPlanner" && r.loss_name == "l2", df)
	df = filter(r -> r.planner == "AStarPlanner" && r.loss_name == "lstar", df)
	ds = compute_stats(df;max_time)
	gdf = DataFrames.groupby(ds, [:domain_name, :arch_name])
	dfs = combine(gdf) do sub_df
		map(unique(sub_df.seed)) do s
			subsub_df = filter(r -> r.seed == s, sub_df)
			i = subsub_df[argmax(collect(zip(subsub_df.tst_solved_fold1, -subsub_df.sol_length_fold1))), :tst_solved_fold2]
			j = subsub_df[argmax(collect(zip(subsub_df.tst_solved_fold2, -subsub_df.sol_length_fold2))), :tst_solved_fold1]
			(i + j) / 2
		end |> mean
	end

	# now, make table from the results
	d = Dict([(r.domain_name, r.arch_name) => r.x1 for r in eachrow(dfs)])
	rows = sort(unique(dfs.domain_name))
	cols = sort(unique(dfs.arch_name))
	data = [get(d, (i,j), missing) for i in rows, j in cols]
	da = DataFrame(hcat(rows, data), vcat(["problem"], cols))
	highlight_table(da; backend, high)
	da
end

function finished(df; max_time = 30)
	# df = filter(r -> r.planner == "AStarPlanner" && r.loss_name == "l2", df)
	df = filter(r -> r.planner == "AStarPlanner" && r.loss_name == "lstar", df)
	ds = compute_stats(df;max_time)
	gdf = DataFrames.groupby(ds, [:domain_name, :arch_name])
	dfs = combine(gdf) do sub_df
		cm = countmap(sub_df.seed)
		join([get(cm,k,0) for k in [1,2,3]],"/")
	end

	# now, make table from the results
	d = Dict([(r.domain_name, r.arch_name) => r.x1 for r in eachrow(dfs)])
	rows = sort(unique(dfs.domain_name))
	cols = sort(unique(dfs.arch_name))
	data = [get(d, (i,j), missing) for i in rows, j in cols]
	da = DataFrame(hcat(rows, data), vcat(["problem"], cols))
	pretty_table(da)
	da
end

function show_vitek()
	max_steps = 10_000
	max_time = 30
	dense_layers = 2
	seed = 1
	dry_run = true
	domain_name = "ferry"
	arch_name = "mixedlrnn2"
	loss_name = "lstar"
	graph_layers = 3
	residual = "none"
	dense_layers = 2
	dense_dim = 8
	seed = 2
	problems = ["blocks","ferry","npuzzle","spanner","elevators_00"]

	cases = vec(collect(Iterators.product(("atombinary","objectatom", "objectbinary"), ("lstar", "l2"), IPC_PROBLEMS, (4, 16, 32), (1, 2, 3), (:none, :linear), (1, 2, 3))))

	# map(shuffle(cases)) do (arch_name, loss_name, domain_name, dense_dim, graph_layers, residual, seed)
	# 	submit_missing(;dry_run, domain_name, arch_name, loss_name, max_steps,  max_time, graph_layers, residual, dense_layers, dense_dim, seed, result_dir = "super_amd_fast")
	# end |> vec |> mean

	df = isfile("super_amd_fast/results.csv") ? CSV.read("super_amd_fast/results.csv", DataFrame) :  DataFrame()

	if !isempty(df)
		done = Set([(r.arch_name, r.loss_name, r.domain_name, r.dense_dim, r.graph_layers, Symbol(r.residual), r.seed,) for r in eachrow(df)])
		cases = filter(e -> e ∉ done, collect(cases))
	end;

	amd_stats = map(cases) do (arch_name, loss_name, domain_name, dense_dim, graph_layers, residual, seed)
		read_data(;domain_name, arch_name, loss_name, max_steps,  max_time, graph_layers, residual, dense_layers, dense_dim, seed, result_dir="super_amd_fast")
	end |> vec;
	amd_stats = filter(!isempty, amd_stats)
	if !isempty(amd_stats)
		dff = reduce(vcat, amd_stats);
		println("adding results from following architectures: ",unique(dff.arch_name))

		df = isempty(df) ? dff : vcat(df, dff);
		CSV.write("super_amd_fast/results.csv", df)
	end

	finished(df);
	vitek_table(df, :tst_solved, highlight_max);
end
show_vitek();