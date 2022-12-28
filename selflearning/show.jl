using Serialization
using Statistics
using DataFrames
using NeuroPlanner
using JSON

###########
#	Collect all stats to one big DataFrame, over which we will perform queries
###########
function extract_stats(e; only_last = false)
	# times = e.times[2:end] .- e.times[1]
	m = length(e.all_solutions)
	ii = only_last ? (m:m) : (1:m)
	map(ii) do i 
		n = length(e.all_solutions[i])
		solved = filter(!isnothing, e.all_solutions[i])
		solved = filter(s -> s.solved, solved)
		l =  filter(x -> x .!= Inf, e.losses[i])
		merge(e.configuration,
		Dict(["sol_time" => round(mean(s.solution_time for s in solved), digits = 2),
			"sol_length" => round(mean(s.sol_length for s in solved), digits = 2),
			"sol_expanded" => round(mean(s.expanded for s in solved), digits = 2),
			"nsolved" => length(solved),
			"solved" => length(solved) / n,
			# "t" => times[i],
			"mean_loss" => mean(l),
			"max_loss" => maximum(l),
		])
		)
	end
end

function extract_stats(problem, number; only_last = false)
	try 
		e = deserialize("results/$(problem)/$(number).jls")
		ds = extract_stats(e; only_last)
		map(d -> merge(d, Dict(["problem" => problem, "configuration" => number])), ds)
	catch 
		println(problem, " ", number)
		return([])
	end
end

problems = ["blocks-slaney", "ferry", "gripper", "n-puzzle"]
df = mapreduce(vcat, problems) do p 
	mapreduce(vcat, 1:10) do i 
		extract_stats(p, i; only_last = true)
	end
end |> DataFrame

CSV.write("results/aggregated.csv", df)

using CSV, DataFrames, HypothesisTests, Statistics
df = CSV.read("results/aggregated.csv", DataFrame)
dff = filter(r -> r.problem == "blocks-slaney", df)

ks = ["sort_by_complexity",
"loss_name",
"solve_solved",
"opt_type",
"artificial_goals",
"graph_dim",
"graph_layers",
"stop_after",
"max_steps",
"double_maxtime",
"planner_name",
"epsilon",
"dense_dim",
"seed",
"dense_layers",
# "max_loss",
"max_time",
]

k = "sort_by_complexity"
scores = df.solved
for k in ks
	label = sort(getproperty(df, k))
	uks = unique(getproperty(df, k))
	length(uks) < 2 && continue
	vals = [scores[label .== u] for u in uks]
	mvals = mean.(vals)
	I = sortperm(mvals)
	uks, vals, mvals = uks[I], vals[I], mvals[I]
	s = mapreduce(*, 1:length(uks) - 1) do i 
		p = pvalue(MannWhitneyUTest(vals[i], vals[i+1]))
		s = (p < 0.05) ? "<" : "≈"
		s*" $(uks[i+1])"
	end
	s = "$(uks[1]) "*s

	println(k,": ", s)
end







function _show_stats(stats)
	solved = filter(!isnothing, stats)
	mean_time = round(mean(s.solution_time for s in solved), digits = 2)
	mean_length = round(mean(s.sol_length for s in solved), digits = 2)
	mean_expanded = round(mean(s.expanded for s in solved), digits = 2)
	mean_excess = round(mean(s.expanded ./ s.sol_length for s in solved), digits = 2)
	println(" solved instances: ", length(solved), 
		" (",round(length(solved) / length(stats), digits = 2), ") mean length ",mean_length, " mean expanded ",mean_expanded, " mean expanded excess ",mean_excess, " mean_time = ", round(mean_time, digits = 3))
end

function average_incomplete_runs(as)
	s = zeros(length(as[1]))
	n = zeros(length(as[1]))
	for a in as 
		for (i,x) in enumerate(a)
			x === missing && continue
			s[i] += x
			n[i] += 1
		end
	end
	n = max.(n,1)
	(s ./ n), s, n
end

function resultfiles(d, prefix)
	filter(endswith(".jls"), filter(startswith(prefix), readdir(d)))
end

# show fraction of solved mazes
function showresults(;index = 1, solve_solved = false)
	df = DataFrame()
	for d in readdir("results")
		for prefix in ["lstar_$(solve_solved)_32", "l2_$(solve_solved)_32",]
			as = map(resultfiles(joinpath("results", d), prefix)) do f 
				s = deserialize(joinpath("results", d, f))
				x = Vector{Union{Float64,Missing}}(missing, 10)
				# x[1:length(s[2])] = map(mean, s[2])
				x[1:length(s.all_solutions)] = map(x -> mean(x .!== nothing), s.all_solutions)
				# x[1:length(s.all_solutions)] = map(x -> mean(i.sol_length for i in x if i !== nothing), s.all_solutions)
				x
			end
			l = first(split(prefix, "_"))
			t = d == "blocks-slaney" ? "blocks" : d
			x = average_incomplete_runs(as)[index]
			x .= round.(x, digits = 2)
			x = map(x -> x == 0 ? "" : x, x)
			df[!,"$(t)_$(l)"] = x
		end
	end 
	df
end


# for d in readdir("results")
#    for f in readdir(joinpath("results", d))
# 	   s = deserialize(joinpath("results", d, f))
# 	   print((d,f, length(s.all_solutions)),": ")
# 	   println(map(x -> mean(x .!== nothing), s.all_solutions))
#    end
# end

