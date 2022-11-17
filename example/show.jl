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

# show fraction of solved mazes
df = DataFrame()
for d in readdir("results")
	for f in ["lstar_false_32_1.jls", "l2_false_32_1.jls",]
		s = deserialize(joinpath("results", d, f))
		l = first(split(f, "_"))
		x = Vector{Union{String,Float64}}(fill("", 10))
		t = d == "blocks-slaney" ? "blocks" : d
		x[1:length(s.all_solutions)] = map(x -> round(mean(x .!== nothing), digits = 2), s.all_solutions)
		df[!,"$(t)_$(l)"] = x
	end
end 
df


for d in readdir("results")
           for f in readdir(joinpath("results", d))
               s = deserialize(joinpath("results", d, f))
               print((d,f),": ")
               println(map(x -> mean(x .!== nothing), s.all_solutions))
           end
       end

