using NeuroPlanner
using PDDL
using Flux
using JSON
using CSV
using SymbolicPlanners
using PDDL: GenericProblem
using SymbolicPlanners: PathSearchSolution
using Statistics
using IterTools
using Random
using StatsBase
using Serialization
using DataFrames
using Mill
using Functors
using Accessors
using Logging
using TensorBoardLogger
using LinearAlgebra
using BenchmarkTools
using Comonicon
using ProfileCanvas
using Profile
using Flux.ChainRulesCore

include("solution_tracking.jl")
include("problems.jl")
include("training.jl")
include("utils.jl")

function graph_stats(kb::KnowledgeBase)
	nv = size(kb[:x1],2)
	ne = mapreduce(ds -> length(ds.data.data[1].data.ii), +, kb[:gnn_2].data)
	[nv, ne]
end

function benchmark_domain_arch(archs, domain_name; difficulty="train")
	graph_layers = 2
	dense_dim = 16
	dense_layers = 2
	residual = "none"

	residual = Symbol(residual)
	domain_pddl, problem_files = getproblem(domain_name, false)
	if difficulty == "train"
		train_files = filter(s -> isfile(plan_file(s)), problem_files)
		train_files = domain_name ∉ IPC_PROBLEMS ? sample(train_files, min(div(length(problem_files), 2), length(train_files)), replace = false) : train_files
		problem_files = train_files
	else
		problem_files = sort(filter(s -> contains(s, difficulty), problem_files))
	end

	domain = load_domain(domain_pddl)

	# function experiment(domain_name, hnet, domain_pddl, train_files, problem_files, filename, fminibatch;max_steps = 10000, max_time = 30, graph_layers = 2, residual = true, dense_layers = 2, dense_dim = 32, settings = nothing)
	models = map(archs) do hnet
		pddld = hnet(domain; message_passes = graph_layers, residual)
		problem = load_problem(first(train_files))
		pddle, state = initproblem(pddld, problem)
		h₀ = pddle(state)
		reflectinmodel(h₀, d -> Dense(d, dense_dim, relu);fsm = Dict("" =>  d -> ffnn(d, dense_dim, 1, dense_layers)))
	end

	map(problem_files[end-5:end]) do problem_file
		problem = load_problem(problem_file)
		state = initstate(domain, problem)
		if difficulty == "train"
			plan = load_plan(problem_file)
			states = SymbolicPlanners.simulate(StateRecorder(), domain, state, plan)
		else

			sol = SymbolicPlanners.solve(AStarPlanner(NullHeuristic(); max_nodes = 10), domain, state, PDDL.get_goal(problem))
		end

		# timing of extraction
		ts = map(archs) do hnet
			pddld = hnet(domain; message_passes = graph_layers, residual)
			pddle, state = initproblem(pddld, problem)
			map(pddle, states)
			mean(@elapsed map(pddle, states) for _ in 1:100) / length(states)
		end


		# timing of extraction + model
		# ts = map(models, archs) do model, hnet
		# 	pddld = hnet(domain; message_passes = graph_layers, residual)
		# 	pddle, state = initproblem(pddld, problem)
		# 	map(model ∘ deduplicate  ∘ pddle, states)
		# 	mean(@elapsed map(model ∘ deduplicate ∘ pddle, states) for _ in 1:100) / length(states)
		# end

		# number of vertices and edges
		# ts = map(archs) do hnet
		# 	pddld = hnet(domain; message_passes = graph_layers, residual)
		# 	pddle, state = initproblem(pddld, problem)
		# 	mean(map(graph_stats ∘ pddle, states))
		# end

		
		ns = tuple([Symbol("$(a)") for a in archs]...)
		stats = merge((;domain_name, problem_file), NamedTuple{ns}(ts))
		@show stats
		stats
	end
end

# archs = [ObjectBinary,ObjectAtom, AtomBinary, ObjectPair]
archs = [ObjectBinary, ObjectBinary2FE, ObjectBinary2FENA, ObjectBinary2ME, ObjectAtom, AtomBinary, AtomBinary2]
data = map(problem -> benchmark_domain_arch(archs, problem), setdiff(IPC_PROBLEMS,["ipc23_sokoban"]))
df = DataFrame(reduce(vcat, data))
gdf = DataFrames.groupby(df, ["domain_name"]);
combine(gdf) do sub_df 
	 (;ObjectBinary = round(1e6*mean(sub_df.ObjectBinary), digits = 1),
	 ObjectBinary2FE = round(1e6*mean(sub_df.ObjectBinary2FE), digits = 1),
	 	ObjectBinary2FENA = round(1e6*mean(sub_df.ObjectBinary2FENA), digits = 1),
	 	ObjectBinary2ME = round(1e6*mean(sub_df.ObjectBinary2ME), digits = 1),
	 	ObjectAtom = round(1e6*mean(sub_df.ObjectAtom), digits = 1), 
	 	AtomBinary = round(1e6*mean(sub_df.AtomBinary), digits = 1),
	 	AtomBinary2 = round(1e6*mean(sub_df.AtomBinary2), digits = 1),
	 )
end


# combine(gdf) do sub_df 
# 	 (;ObjectBinary = tuple(round.(mean(sub_df.ObjectBinary), digits = 1)...),
# 	 	ObjectAtom = tuple(round.(mean(sub_df.ObjectAtom), digits = 1)...),
# 	 	AtomBinary = tuple(round.(mean(sub_df.AtomBinary), digits = 1)...),
# 	 	AtomBinary2 = tuple(round.(mean(sub_df.AtomBinary2), digits = 1)...),
# 	 )
# end


#######
#	Average number of vertices and edges per problem
#######
 Row │ domain_name        ObjectBinary   ObjectAtom     AtomBinary        AtomBinary2
     │ String             Tuple…         Tuple…         Tuple…            Tuple…
─────┼─────────────────────────────────────────────────────────────────────────────────────
   1 │ ipc23_ferry        (23.8, 24.4)   (23.8, 24.4)   (26.4, 128.7)     (26.4, 83.4)
   2 │ ipc23_rovers       (35.0, 459.2)  (35.0, 238.7)  (261.1, 22204.2)  (261.1, 21620.9)
   3 │ ipc23_blocksworld  (17.7, 25.3)   (17.7, 25.3)   (45.4, 272.6)     (45.4, 209.1)
   4 │ ipc23_floortile    (34.0, 139.0)  (34.0, 139.0)  (156.7, 3032.1)   (156.7, 2764.5)
   5 │ ipc23_satellite    (42.0, 102.5)  (42.0, 102.5)  (112.9, 1627.6)   (112.9, 1425.3)
   6 │ ipc23_spanner      (28.0, 27.0)   (28.0, 27.0)   (46.3, 225.5)     (46.3, 155.2)
   7 │ ipc23_childsnack   (26.5, 8.4)    (26.5, 8.4)    (28.1, 90.2)      (28.1, 56.9)
   8 │ ipc23_miconic      (29.8, 200.9)  (29.8, 200.9)  (217.8, 7726.0)   (217.8, 7310.2)
   9 │ ipc23_transport    (25.2, 73.0)   (25.2, 73.0)   (73.0, 1522.2)    (73.0, 1391.6)




# EdgeBuilder on M3 in μs with a HMGNNnetwork with config (graph_layers = 2, dense_dim = 16, dense_layers = 2, residual = "none")
# without state deduplication
#  Row │ domain_name        ObjectBinary  ObjectAtom  AtomBinary  AtomBinary2
#      │ String             Float64       Float64     Float64     Float64
# ─────┼──────────────────────────────────────────────────────────────────────
#    1 │ ipc23_ferry                97.5        56.0       135.1        119.8
#    2 │ ipc23_rovers             3759.7      3876.5      4725.7       3987.5
#    3 │ ipc23_blocksworld          79.9        67.3       210.1        195.0
#    4 │ ipc23_floortile           363.1       305.5      1046.1        798.1
#    5 │ ipc23_satellite           528.1       262.6       619.1        497.6
#    6 │ ipc23_spanner             172.2       108.7       175.4        164.5
#    7 │ ipc23_childsnack          169.0       105.9       123.3        102.0
#    8 │ ipc23_miconic             188.8       149.4      1928.4       1514.2
#    9 │ ipc23_transport           283.9       194.8       500.4        451.2

# EdgeBuilder on M3 in μs with a HMGNNnetwork with config (graph_layers = 2, dense_dim = 16, dense_layers = 2, residual = "none")
   # with state deduplication
 Row │ domain_name        ObjectBinary  ObjectAtom  AtomBinary  AtomBinary2
     │ String             Float64       Float64     Float64     Float64
─────┼──────────────────────────────────────────────────────────────────────
   1 │ ipc23_ferry                58.0        55.9       119.2        111.3
   2 │ ipc23_rovers              596.8       614.6      4695.1       3941.8
   3 │ ipc23_blocksworld          69.6        66.4       212.0        202.5
   4 │ ipc23_floortile           308.6       299.6       997.6        764.8
   5 │ ipc23_satellite           224.4       232.2       590.9        471.2
   6 │ ipc23_spanner             109.4        97.3       176.5        167.7
   7 │ ipc23_childsnack          102.5        93.3       124.8        103.5
   8 │ ipc23_miconic             149.2       146.9      1884.5       1486.6
   9 │ ipc23_transport           185.0       181.5       507.1        447.3



# f0a267e17a8bce06bd0f88233902279471f6f605
# EdgeBuilder on M3 in μs
#  Row │ domain_name        ObjectBinary  ObjectBinary2  ObjectAtom  AtomBinary  AtomBinary2
#      │ String             Float64       with agg.         Float64     Float64     Float64
# ─────┼─────────────────────────────────────────────────────────────────────────────────────
#    1 │ ipc23_ferry                 8.7            9.4         6.6        23.2         21.2
#    2 │ ipc23_rovers               86.0           59.3        87.8       990.0        402.0
#    3 │ ipc23_blocksworld          11.2           11.3         6.9        33.6         28.5
#    4 │ ipc23_floortile            41.5           28.5        34.9       300.3        123.9
#    5 │ ipc23_satellite            32.4           22.9        29.8       196.1         82.4
#    6 │ ipc23_spanner              19.6           13.7        13.7        37.5         31.0
#    7 │ ipc23_childsnack           18.0           11.3        12.6        23.2         19.0
#    8 │ ipc23_miconic              41.1           37.4        38.5       520.2        210.5
#    9 │ ipc23_transport            23.5           18.2        22.0        90.5         60.6
