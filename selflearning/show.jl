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
problem="npuzzle"
number=1
loss = "lstar"
function parse_results(problem, loss; testset = true)
	numbers = filter(1:3) do number
		isfile("supervised/$(problem)/$(loss)_40000_30_2_8_2_32_$(number).jls")
	end
	if isempty(numbers) 
		n = (Symbol("$(loss)_astar"), Symbol("$(loss)_gbfs"))
		return(NamedTuple{n}((NaN,NaN)))
	end
	x = map(numbers) do number
		filename = "supervised/$(problem)/$(loss)_40000_30_2_8_2_32_$(number).jls"
		stats = deserialize(filename)
		df = DataFrame(stats[:])
		da = filter(df) do r 
			(r.used_in_train !== testset) && (r.planner == "AStarPlanner")
		end
		dg = filter(df) do r 
			(r.used_in_train !== testset) && (r.planner == "GreedyPlanner")
		end
		# [mean(da.sol_length), mean(dg.sol_length)]
		# [mean(da.solved), mean(dg.solved)]
		[mean(da.expanded), mean(dg.expanded)]
	end |> mean
	# x = round.(x, digits = 2)
	x = round.(x, digits = 0)
	n = (Symbol("$(loss)_astar"), Symbol("$(loss)_gbfs"))
	NamedTuple{n}(tuple(x...))
end

let 
	problems = ["blocks", "ferry", "gripper", "npuzzle"]
	df = map(problems) do problem
		mapreduce(merge,["lstar","lgbfs","lrt","l2"]) do loss
			parse_results(problem, loss;testset = true)
		end
	end |> DataFrame;
	hcat(DataFrame(problem = problems),  df[:,1:2:end], df[:,2:2:end])
end
