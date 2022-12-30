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

function show(problems = ["blocks-slaney", "ferry", "gripper", "n-puzzle"])
	df = mapreduce(vcat, problems) do p 
		mapreduce(vcat, 1:100) do i 
			extract_stats(p, i; only_last = true)
		end
	end |> DataFrame

	# CSV.write("results/aggregated.csv", df)
	# df = CSV.read("results/aggregated.csv", DataFrame)
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
	map(ks) do k
		label = sort(getproperty(df, k))
		uks = unique(getproperty(df, k))
		length(uks) < 2 && return(k,"$(uks[1]) ")
		vals = [scores[label .== u] for u in uks]
		mvals = mean.(vals)
		I = sortperm(mvals)
		uks, vals, mvals = uks[I], vals[I], mvals[I]
		s = mapreduce(*, 1:length(uks) - 1) do i 
			p = pvalue(MannWhitneyUTest(vals[i], vals[i+1]))
			s = (p < 0.05) ? "<" : "≈"
			s*" $(uks[i+1])($(length(vals[i+1]))) "
		end
		s = "$(uks[1])($(length(vals[1]))) "*s
		# println(k,": ", s)
		(k,s)
	end |> DataFrame
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
function showresults()
	df = DataFrame()
	for d in readdir("deepcubea2")
		# for prefix in ["lstar_mean_0.5_30_2_8_2_32_10000_100000_0.0_1:30_1.0", "l2_mean_0.5_30_2_8_2_32_10000_100000_0.0_1:30_1.0",]
		for prefix in ["lstar_mean_0.5_30_2_8_2_32_1000_10000_0.0_1:30_1.0", "l2_mean_0.5_30_2_8_2_32_1000_10000_0.0_1:30_1.0",]
			as = map(resultfiles(joinpath("deepcubea2", d), prefix)) do f 
				stats = deserialize(joinpath("deepcubea2", d, f))
				map(x -> mean(x.train_solutions .!== nothing), stats)
			end
			isempty(as) && continue
			x = mean(as)
			x .= round.(x, digits = 2)
			l = first(split(prefix,"_"))
			xx = zeros(4)
			xx[1:length(x)] .= x
			df[!,"$(d)_$(l)"] = xx
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

