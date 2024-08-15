using Serialization
using Statistics
using DataFrames
using NeuroPlanner
using CSV
using DataFrames
using Random
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

"""

"""
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
	     	tst_solved_sum_fold1 = sum(solved[i1]),
	     	tst_solved_sum_fold2 = sum(solved[i2]),
	     	sol_length_fold1 = sum(sub_df.sol_length[i1]),
	     	sol_length_fold2 = sum(sub_df.sol_length[i2]),
		)
	end
end

function submit_missing(;dry_run = true, domain_name, arch_name, loss_name, max_steps,  max_time, graph_layers, aggregation, residual, dense_layers, dense_dim, seed, result_dir = "super")
	filename = joinpath(result_dir, domain_name, join([arch_name, loss_name, max_steps,  max_time, graph_layers, aggregation, residual, dense_layers, dense_dim, seed], "_"))
	if isfile(filename*"_stats.jls")
		println(@green "finished stats "*filename)
	 	return(:finished)
	end
	if isfile(filename*"_model.jls") && isfile(filename*"_stats_tmp.jls")
		println(@cyan "finished model with temporary stats "*filename)
		dry_run || run(`sbatch -D /home/pevnytom/julia/Pkg/NeuroPlanner.jl/selflearning slurm.sh $domain_name $dense_dim $graph_layers $residual $loss_name $arch_name $seed`)
	 	return(:partial)
	end

	if isfile(filename*"_model.jls")
		println(@yellow "finished model "*filename)
		dry_run || run(`sbatch -D /home/pevnytom/julia/Pkg/NeuroPlanner.jl/selflearning slurm.sh $domain_name $dense_dim $graph_layers $residual $loss_name $arch_name $seed`)
	 	return(:model)
	end
	println(@red "submit "*filename)
	dry_run || run(`sbatch -D /home/pevnytom/julia/Pkg/NeuroPlanner.jl/selflearning slurm.sh $domain_name $dense_dim $graph_layers $residual $loss_name $arch_name $seed`)
	return(:nothing)
end

function read_data(;domain_name, arch_name, loss_name, max_steps, aggregation,  max_time, graph_layers, residual, dense_layers, dense_dim, seed, result_dir="super", suffix = "_stats.jls")
	filename = joinpath(result_dir, domain_name, join([arch_name, loss_name, max_steps,  max_time, graph_layers, aggregation, residual, dense_layers, dense_dim, seed], "_"))
	filename = filename*suffix
	!isfile(filename) && return(DataFrame())
	stats = deserialize(filename)
	df = stats isa DataFrame ? stats : DataFrame(vec(stats))
	select!(df, Not(:trajectory))
	df[:,:domain_name] .= domain_name
	df[:,:arch_name] .= arch_name
	df[:,:loss_name] .= loss_name
	df[:,:graph_layers] .= graph_layers
	df[:,:dense_dim] .= dense_dim
	df[:,:aggregation] .= aggregation
	df[:,:dense_layers] .= dense_layers
	df[:,:residual] .= residual
	df[:,:seed] .= seed
	df[:,:max_steps] .= max_steps
	df[:,:max_time] .= max_time
	df
end

function filter_results(df; planner = nothing, loss_name = nothing, max_time = nothing)
	if planner !== nothing
		df = filter(r -> r.planner == planner, df);
	end

	if loss_name !== nothing
		df = filter(r -> r.loss_name == loss_name, df);
	end

	if max_time !== nothing
		df = filter(r -> r.max_time == max_time, df);
	end
	df
end

"""
	coverage_table(df, k, high; max_time = 30)

	compute coverage on testing set while using half of test as validation
"""
function coverage_table(df, k, high; max_time = 30)
	ds = compute_stats(df; max_time)
	gdf = DataFrames.groupby(ds, [:domain_name, :arch_name])
	dfs = combine(gdf) do sub_df
		map(unique(sub_df.seed)) do s
			subsub_df = filter(r -> r.seed == s, sub_df)
			i = subsub_df[argmax(collect(zip(subsub_df.tst_solved_fold1, -subsub_df.sol_length_fold1))), :tst_solved_fold2]
			j = subsub_df[argmax(collect(zip(subsub_df.tst_solved_fold2, -subsub_df.sol_length_fold2))), :tst_solved_fold1]
			(i + j) / 2
		end |> skipmissing |> mean
	end

	# now, make table from the results
	d = Dict([(r.domain_name, r.arch_name) => r.x1 for r in eachrow(dfs)])
	rows = sort(unique(dfs.domain_name))
	cols = sort(unique(dfs.arch_name))
	data = [get(d, (i,j), missing) for i in rows, j in cols]
	da = DataFrame(hcat(rows, data), vcat(["problem"], cols))
	# highlight_table(da; backend, high)
	da
end

function best_configuration(df)
	stats = compute_stats(df)
	gdf = DataFrames.groupby(stats, [:domain_name, :arch_name])
	dfs = combine(gdf) do sub_df
		i = argmax(sub_df.tst_solved)
		sub_df[i,:]
	end

	# now, make table from the results
	d = Dict([(r.domain_name, r.arch_name) => join([r.dense_dim, r.graph_layers]," / ") for r in eachrow(dfs)])
	rows = sort(unique(dfs.domain_name))
	cols = sort(unique(dfs.arch_name))
	data = [get(d, (i,j), missing) for i in rows, j in cols]
	da = DataFrame(hcat(rows, data), vcat(["problem"], cols))
	highlight_table(da; backend, high)
	da
end

"""
	paper_table(df; highlighter = highlight_max, add_rank = true)

	format the results for the paper, mainly ordering of the columns
"""
function paper_table(df; highlighter = highlight_max, add_rank = true)
	#add rank
	if add_rank
		rank = map(eachrow(df)) do row
			r = [ismissing(r) ? 0 : r for r in row[2:end]]
			tiedrank(r,rev=true)
		end |> mean
		push!(df, vcat(["mean rank"],rank))
	end

	blank_col = fill(missing, size(df,1))
	problems = map(s -> replace(s, "ipc23_" => ""), df.problem)
	data = hcat(problems, blank_col,
			df.objectbinaryfe, df.objectbinaryfena, df.objectbinaryme, blank_col, 
			df.atombinaryfe, df.atombinaryfena, df.atombinaryme, blank_col,
			df.objectatombipfe, df.objectatombipfena, df.objectatombipme, df.objectatom, blank_col,
			 df.asnet, df.hgnn)
	data = map(x -> x isa Number ? round(100 * x, digits = 1) : x, data)
	header = ["problem","",
		"ObjectBinaryFE", "ObjectBinaryfena", "ObjectBinaryme", "", 
		"AtomBinaryfe", "AtomBinaryfena", "AtomBinaryme", "",
			"ObjectAtomBipfe", "ObjectAtomBipfena", "ObjectAtomBipme", "ObjectAtom", "",
			 "ASNet", "HGNN"]

	h = [highlighter(data, i, j) for i in 1:size(data,1), j in 1:size(data,2)]
	h[h .=== missing] .= false
	# after highlighting remove missings
	data[:,[2,6,10,15]] .= ""

	# text backend
	highlighters = Highlighter((data, i, j) -> h[i, j], crayon"yellow")
	pretty_table(data ; header, backend = Val(:text) , highlighters)
	# latex backend
	highlighters = LatexHighlighter((data, i, j) -> h[i, j], ["textbf"])
	pretty_table(data ; header, backend = Val(:latex) , highlighters)
end

function finished(df; max_time = 30)
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

"""
	find_smallest_solutions(df)

	find smallest plans from all soutions
"""
function find_smallest_solutions(df)
	# first, go over all files and identify shortes trajectories
	stat_files = map(walkdir("super_amd_gnn")) do (root, dirs, files)
		stat_files = filter(file -> endswith(file, "_stats.jls"), files)
		map(s -> joinpath(root, s), stat_files)
	end
	stat_files = reduce(vcat, stat_files)
	df = map(stat_files) do f
		stats = deserialize(f)
		stats = filter(s -> s.solved, stats)
		stats = stats[:,[:problem_file, :trajectory, :sol_length]]
		stats[:,:stat_file] .= f
		stats
	end

	gdf = DataFrames.groupby(dff, [:problem_file])
	combine(gdf) do sub_df 
		i = argmin(sub_df.sol_length)
		sub_df[i,:]
	end
end


"""
	dataframe_for_pairplots(df)

	filter the dataset to keep only results of the best configuration for triplets 
	[:domain_name, :arch_name, :seed].
	This is used to produce dataframe used to plot Figures 5 and 6 in the paper.
"""
function dataframe_for_pairplots(df)
	function select_val_return_test(df, val_ii, tst_ii)
		performance = combine(DataFrames.groupby(df, [:graph_layers, :residual, :dense_layers, :dense_dim])) do sub_df
			dff = filter(r -> r.problem_file ∈ val_ii, sub_df)
			(;coverage = mean(dff.solved))
		end
		best = performance[argmax(performance.coverage),:]
		filter(r -> r.graph_layers == best.graph_layers && r.residual == best.residual && r.dense_layers == best.dense_layers && r.dense_dim == best.dense_dim && r.problem_file ∈ tst_ii, df)
	end
	
	gdf = DataFrames.groupby(df, [:domain_name, :arch_name, :seed])
	dff = combine(gdf) do sub_df
		ii = sort(unique(sub_df.problem_file))
		i1 = ii[1:2:end]
		i2 = ii[2:2:end]
		vcat(
			select_val_return_test(sub_df, i1, i2),
			select_val_return_test(sub_df, i2, i1),
		)
	end
end

"""
	add_temporary_results(df, cases; result_dir="super_amd_gnn")

	Add results from the experiments which did not finish in time, yet they still could solve some stuff.
	The difficulty is that we need to add "bogus" information about results on unfinished problem instances.
	Note that the `used_in_train` field is slightly wrong, since it is set to `false` for all unfinished results.
"""
function add_temporary_results(df, cases; result_dir="super_amd_gnn")
	empty_fields = Dict()
	problem_files = map(IPC_PROBLEMS) do domain_name
		problem_files = getproblem(domain_name, false)[2]
		domain_name => problem_files
	end |> Dict
	new_dfs = map(cases) do (arch_name, loss_name, domain_name, dense_dim, graph_layers, aggregation, residual, seed)
		new_df = read_data(;domain_name, arch_name, loss_name, max_steps, max_time, graph_layers, aggregation, residual, dense_layers, dense_dim, seed, result_dir, suffix = "_stats_tmp.jls")		

		# We need to construct new bogus fields marking that the attempt has failed
		bogus_stats = get!(empty_fields, domain_name) do
			fdf = filter(r -> r.domain_name == domain_name, df)
			(domain_name,
			arch_name,
			loss_name,
			graph_layers,
			dense_dim,
			aggregation,
			dense_layers,
			residual,
			seed,
			max_steps,
			max_time,
			solution_time = typemax(Float64),
			sol_length = typemax(Int64),
			expanded = typemax(Int64),
			solved = false,
			used_in_train = false,
			time_in_heuristic = typemax(Float64),
			)
		end

		_problem_files = isempty(new_df) ? problem_files[domain_name] : setdiff(problem_files[domain_name], filter(r -> r.planner == "AStarPlanner", new_df).problem_file)
		for problem_file in _problem_files
			push!(new_df, merge(bogus_stats, (;problem_file, planner = "AStarPlanner")))
		end

		_problem_files = isempty(new_df) ? problem_files[domain_name] : setdiff(problem_files[domain_name], filter(r -> r.planner == "GreedyPlanner", new_df).problem_file)
		for problem_file in _problem_files
			push!(new_df, merge(bogus_stats, (;problem_file, planner = "GreedyPlanner")))
		end
		new_df
	end

	new_dfs = reduce(vcat, new_dfs)
	vcat(df, new_dfs)
end

"""
	submit_experiments()

	submit all experiments required for the results, except the experiments with LAMA planner
"""
function submit_experiments()
	max_steps = 10_000
	max_time = 30
	dense_layers = 2
	dry_run = true
	all_archs = ["objectbinaryme", "objectbinaryfena", "atombinaryme", "atombinaryfena", "objectatom", "objectatombipfe",  "objectatombipfena", "atombinaryfe", "objectbinaryfe", "objectatombipme","asnet", "hgnn",]

	cases = vec(collect(Iterators.product(all_archs, ("l2", "lstar",), IPC_PROBLEMS, (4, 16, 32), (1, 2, 3), ("summax",), (:none, ), (1, 2, 3))))

	map(cases) do (arch_name, loss_name, domain_name, dense_dim, graph_layers, aggregation,  residual, seed)
		submit_missing(;dry_run, domain_name, arch_name, aggregation, loss_name, max_steps,  max_time, graph_layers, residual, dense_layers, dense_dim, seed, result_dir = "super_amd_gnn")
	end |> vec |> countmap

	max_steps = 10_000
	max_time = 180
	dense_layers = 2
	dry_run = true
	all_archs = ["objectbinaryfe"]

	cases = vec(collect(Iterators.product(all_archs, ("lstar",), IPC_PROBLEMS, (4, 16, 32), (1, 2, 3), ("summax",), (:none, ), (1, 2, 3))))

	map(cases) do (arch_name, loss_name, domain_name, dense_dim, graph_layers, aggregation,  residual, seed)
		submit_missing(;dry_run, domain_name, arch_name, aggregation, loss_name, max_steps,  max_time, graph_layers, residual, dense_layers, dense_dim, seed, result_dir = "super_amd_gnn")
	end |> vec |> countmap
end

function create_tables()
	max_steps = 10_000
	max_time = 30
	dense_layers = 2
	dry_run = true
	all_archs = ["objectbinaryme", "objectbinaryfena", "atombinaryme", "atombinaryfena", "objectatom", "objectatombipfe",  "objectatombipfena", "atombinaryfe", "objectbinaryfe", "objectatombipme","asnet", "hgnn",]

	cases = vec(collect(Iterators.product(all_archs, ("l2", "lstar",), IPC_PROBLEMS, (4, 16, 32), (1, 2, 3), ("summax",), (:none, ), (1, 2, 3))))

	# loads cached results
	df = isfile("super_amd_gnn/results.csv") ? CSV.read("super_amd_gnn/results.csv", DataFrame) :  DataFrame();

	# add new finished results
	if !isempty(df)
		done = Set([(r.arch_name, r.loss_name, r.domain_name, r.dense_dim, r.graph_layers, r.aggregation, Symbol(r.residual), r.seed,) for r in eachrow(df)])
		cases = filter(e -> e ∉ done, collect(cases))
	end;

	amd_stats = map(cases) do (arch_name, loss_name, domain_name, dense_dim, graph_layers, aggregation, residual, seed)
		read_data(;domain_name, arch_name, loss_name, max_steps, max_time, graph_layers, aggregation, residual, dense_layers, dense_dim, seed, result_dir="super_amd_gnn")
	end |> vec;
	amd_stats = filter(!isempty, amd_stats)
	if !isempty(amd_stats)
		dff = reduce(vcat, amd_stats);
		println("adding results from following architectures: ",unique(dff.arch_name))
		df = isempty(df) ? dff : vcat(df, dff);
		CSV.write("super_amd_gnn/results.csv", df)
	end


	# this adds the results which has not finished in time
	df = add_temporary_results(df, cases; result_dir="super_amd_gnn")
	# this removes all training problem instances, especially those not used for training (due to missing plan)
	df = filter(r -> !(contains(r.problem_file, "training") && !r.used_in_train), df);


	# print the coverage table of lstar with AStarPlanner and max_time 30 (Table 1 in the paper)
	dff = filter_results(df; planner = "AStarPlanner", loss_name = "lstar", max_time = 30)
	paper_table(coverage_table(dff, :tst_solved, highlight_max; max_time, add_rank = false))
end