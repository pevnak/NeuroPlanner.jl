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

function padded_mean(xs; l = maximum(length.(xs)))
	# pad with the last value
	xs = map(xs) do x 
		length(x) > l && return(x[1:l])
		length(x) == l && return(x)
		vcat(x, fill(x[end], l - length(x)))
	end
	mean(xs)
end

function read_data(problem, loss; l = 10)
	files = filter(s -> contains(s, loss) && endswith(s, "_stats.jls"), readdir("selflearning/$(problem)/"))
	isempty(files) && return(fill(missing, l))
	xs = map(files) do f 
		stats = deserialize("selflearning/$(problem)/$(f)")[2]
		map(o -> mean([x !== nothing for x in o]), stats)
	end
	round.(Int, 100*padded_mean(xs; l))
end


# problems = filter(s -> isdir("selflearning",s), readdir("selflearning"))
problems = ["blocks", "elevators_00", "ferry", "gripper", "npuzzle", "spanner"]
data = map(Iterators.product(["lstar","l2", "levin"], problems)) do (loss_name, problem)
	Symbol("$(problem) $(loss_name)") => read_data(problem, loss_name)
end |> vec |> DataFrame

stats = map(Iterators.product(["lstar","l2", "levin"], problems)) do (loss_name, problem)
	(problem, loss_name) => read_data(problem, loss_name)
end |> vec
header = ([x[1][1] for x in stats], [x[1][2] for x in stats])
data = hcat([x[2] for x in stats]...)


function high(data, i, j)
	k = (j - 1) ÷ 3
	s = argmax(data[i,3*k+s] for s in 1:3)
	s == mod(j - 1, 3) + 1
end

pretty_table(data; header, backend = Val(:text), highlighters = (Highlighter(high, crayon"yellow"),))
pretty_table(data, backend = Val(:latex), highlighters = (LatexHighlighter(high, "textbf"),))
