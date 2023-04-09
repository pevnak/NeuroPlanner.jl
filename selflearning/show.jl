using Serialization
using Statistics
using DataFrames
using NeuroPlanner
using JSON
using CSV
using DataFrames
using HypothesisTests
using Statistics

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

max_steps = 10_000
max_time = 30
dense_layers = 2
seed = 1
problems = ["blocks","ferry","npuzzle","gripper"]
stats = map(Iterators.product(("asnet","pddl"), ("lstar","l2"), problems, (4, 8, 16), (1, 2, 3), (:none, :linear, :dense))) do (arch_name, loss_name, domain_name, dense_dim, graph_layers, residual)
	read_data(;domain_name, arch_name, loss_name, max_steps,  max_time, graph_layers, residual, dense_layers, dense_dim, seed)
end
df = reduce(vcat, filter(!isempty, vec(stats)))

gdf = DataFrames.groupby(df, [:domain_name, :arch_name, :loss_name, :max_steps,  :max_time, :graph_layers, :residual, :dense_layers, :dense_dim, :seed])
stats = combine(gdf) do sub_df
	(;	trn_solved = mean(sub_df.solved[sub_df.used_in_train]),
     	tst_solved = mean(sub_df.solved[.!sub_df.used_in_train]),
	)
end

dfs = map(unique(df.loss_name)) do ln 
	df1 = filter(r -> r.loss_name == ln, df)
	dfs = map(unique(df1.arch_name)) do an 
		df2 = filter(r -> r.arch_name == an, df1)
		gdf = DataFrames.groupby(stats, :domain_name)
		a = combine(gdf) do sub_df
			# (;average = mean(sub_df.tst_solved), maximum = maximum(sub_df.tst_solved))
			DataFrame("$ln $an" => [maximum(df2.tst_solved)])
		end
	end
	leftjoin(dfs..., on = :domain_name)
end
leftjoin(dfs..., on = :domain_name)

# investigate 

gdf = DataFrames.groupby(stats, [:domain_name, :graph_layers])
a = combine(gdf) do sub_df
	(;average = mean(sub_df.tst_solved), maximum = maximum(sub_df.tst_solved))
end

gdf = DataFrames.groupby(stats, [:domain_name, :residual])
a = combine(gdf) do sub_df
	(;average = mean(sub_df.tst_solved), maximum = maximum(sub_df.tst_solved))
end



