using Serialization
using Statistics
using DataFrames
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

