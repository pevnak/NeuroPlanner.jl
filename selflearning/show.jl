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
loss_name = "lstar"
max_steps = 20_000
max_time = 30
graph_layers = 1
dense_dim = 32
dense_layers = 2
residual = "none"
seed = 1

function read_data(;domain_name, loss_name, max_steps,  max_time, graph_layers, residual, dense_layers, dense_dim, seed)
	filename = joinpath("superhyper", domain_name, join([loss_name, max_steps,  max_time, graph_layers, residual, dense_layers, dense_dim, seed], "_")*".jls")
	!isfile(filename) && return(DataFrame())
	df = DataFrame(vec(deserialize(filename)))
	select!(df, Not(:trajectory))
	df[:,:domain_name] .= domain_name
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

problems = ["blocks","ferry","npuzzle","gripper"]
stats = map(Iterators.product(problems, (4, 8, 16, 32), (1, 2, 3), (:none, :linear, :dense))) do (domain_name, dense_dim, graph_layers, residual)
	read_data(;domain_name, loss_name, max_steps,  max_time, graph_layers, residual, dense_layers, dense_dim, seed)
end
df = reduce(vcat, filter(!isempty, vec(stats)))

gdf = DataFrames.groupby(df, [:domain_name, :loss_name, :max_steps,  :max_time, :graph_layers, :residual, :dense_layers, :dense_dim, :seed])
stats = combine(gdf) do sub_df
	(;	trn_solved = mean(sub_df.solved[sub_df.used_in_train]),
     	tst_solved = mean(sub_df.solved[.!sub_df.used_in_train]),
	)
end

gdf = DataFrames.groupby(stats, :domain_name)
a = combine(gdf) do sub_df
	i = argmax(sub_df.tst_solved)
	sub_df[i,:]
end;

a[:,[:domain_name, :tst_solved, :trn_solved, :dense_dim, :graph_layers, :residual, :dense_layers]]


# investigate 
gdf = DataFrames.groupby(stats, [:domain_name, :dense_dim])
a = combine(gdf) do sub_df
	(;average = mean(sub_df.tst_solved), maximum = maximum(sub_df.tst_solved))
end

gdf = DataFrames.groupby(stats, [:domain_name, :graph_layers])
a = combine(gdf) do sub_df
	(;average = mean(sub_df.tst_solved), maximum = maximum(sub_df.tst_solved))
end

gdf = DataFrames.groupby(stats, [:domain_name, :residual])
a = combine(gdf) do sub_df
	(;average = mean(sub_df.tst_solved), maximum = maximum(sub_df.tst_solved))
end



