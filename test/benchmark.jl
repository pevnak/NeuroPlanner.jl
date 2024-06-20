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
		# 	map(model ∘ pddle, states)
		# 	mean(@elapsed map(model ∘ pddle, states) for _ in 1:100) / length(states)
		# end

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

archs = [ObjectBinaryFE, ObjectBinaryFENA, ObjectBinaryME, ObjectAtom, ObjectAtomBipFE, ObjectAtomBipFENA, ObjectAtomBipME, AtomBinaryFE, AtomBinaryFENA, AtomBinaryME, ASNet, HGNN]
data = map(problem -> benchmark_domain_arch(archs, problem), setdiff(IPC_PROBLEMS,["ipc23_sokoban"]))
df = DataFrame(reduce(vcat, data))
gdf = DataFrames.groupby(df, ["domain_name"]);
combine(gdf) do sub_df 
	 (ObjectBinaryFE = round(1e6*mean(sub_df.ObjectBinaryFE), digits = 1),
	 	ObjectBinaryFENA = round(1e6*mean(sub_df.ObjectBinaryFENA), digits = 1),
	 	ObjectBinaryME = round(1e6*mean(sub_df.ObjectBinaryME), digits = 1),
	 	ObjectAtom = round(1e6*mean(sub_df.ObjectAtom), digits = 1), 
	 	ObjectAtomBipFE = round(1e6*mean(sub_df.ObjectAtomBipFE), digits = 1), 
	 	ObjectAtomBipFENA = round(1e6*mean(sub_df.ObjectAtomBipFENA), digits = 1), 
	 	ObjectAtomBipME = round(1e6*mean(sub_df.ObjectAtomBipME), digits = 1), 
	 	AtomBinaryFE = round(1e6*mean(sub_df.AtomBinaryFE), digits = 1), 
	 	AtomBinaryFENA = round(1e6*mean(sub_df.AtomBinaryFENA), digits = 1), 
	 	AtomBinaryME = round(1e6*mean(sub_df.AtomBinaryME), digits = 1), 
	 	ASNet = round(1e6*mean(sub_df.ASNet), digits = 1), 
	 	HGNN = round(1e6*mean(sub_df.HGNN), digits = 1), 
	 )
end

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
#  Row │ domain_name        ObjectBinaryFE  ObjectBinaryFENA  ObjectBinaryME  ObjectAtom  ObjectAtomBipFE  ObjectAtomBipFENA  ObjectAtomBipME  AtomBinaryFE  AtomBinaryFENA  AtomBinaryME
#      │ String             Float64         Float64           Float64         Float64     Float64          Float64            Float64          Float64       Float64         Float64
# ─────┼──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
#    1 │ ipc23_ferry                  22.2              25.1            31.7        31.2             41.1               38.0             50.6          45.3            45.3          70.3
#    2 │ ipc23_rovers                122.9             231.3           480.8       442.5            386.9              371.8            435.0        6225.4          6417.8        6743.3
#    3 │ ipc23_blocksworld            26.5              25.5            34.3        32.0             45.7               41.5             57.8          90.3            94.0         129.6
#    4 │ ipc23_floortile              78.4              74.8           211.1       218.5            158.4              147.3            174.4        1039.6          1007.4         916.2
#    5 │ ipc23_satellite              63.6              62.4           173.0       171.6            121.6              115.8            135.5         598.5           580.9         503.7
#    6 │ ipc23_spanner                31.0              32.3            74.4        78.4             50.4               49.3             62.2          78.4            75.0         120.1
#    7 │ ipc23_childsnack             25.1              21.8            71.6        63.0             32.4               26.2             46.0          42.5            48.2          71.8
#    8 │ ipc23_miconic               103.4              92.6           130.1       135.6            201.4              187.7            218.1        2292.2          2166.9        2316.5
#    9 │ ipc23_transport              46.7              48.5           120.0       122.0             87.7               82.9            101.1         535.2           527.7         409.0


# EdgeBuilder on M3 in μs with a HMGNNnetwork with config (graph_layers = 2, dense_dim = 16, dense_layers = 2, residual = "none")
   # with state deduplication
#  Row │ domain_name        ObjectBinaryFE  ObjectBinaryFENA  ObjectBinaryME  ObjectAtom  ObjectAtomBipFE  ObjectAtomBipFENA  ObjectAtomBipME  AtomBinaryFE  AtomBinaryFENA  AtomBinaryME
#      │ String             Float64         Float64           Float64         Float64     Float64          Float64            Float64          Float64       Float64         Float64
# ─────┼──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
#    1 │ ipc23_ferry                  46.4              43.1            60.0        59.3             56.7               53.9             71.0          67.4            63.8         100.0
#    2 │ ipc23_rovers                158.2             235.1           529.9       642.5            334.5              325.3            468.9        4890.7          4854.8        3914.0
#    3 │ ipc23_blocksworld            50.4              52.9            73.2        69.5             65.1               62.8             85.6         110.8           105.1         190.8
#    4 │ ipc23_floortile              95.6              95.9           260.8       304.9            162.2              152.3            208.6         692.3           651.3         729.9
#    5 │ ipc23_satellite              97.7              85.2           198.5       234.4            137.9              117.5            150.2         426.7           376.1         440.3
#    6 │ ipc23_spanner                51.8              47.5            85.1       113.4             59.0               56.9             93.0          83.4           101.4         158.1
#    7 │ ipc23_childsnack             47.5              46.6            83.7       100.8             53.5               56.5             62.5          71.4            60.8         101.1
#    8 │ ipc23_miconic                97.7              95.0           131.6       147.6            204.4              194.9            225.0        1740.1          1541.6        1385.5
#    9 │ ipc23_transport              74.6              61.9           158.9       183.8            100.9               95.6            124.2         401.2           377.9         436.4



# f0a267e17a8bce06bd0f88233902279471f6f605
# EdgeBuilder on M3 in μs
# The adventage of FE over ME depends on number of predicates types with higher arities
#  Row │ domain_name        ObjectBinaryFE  ObjectBinaryFENA  ObjectBinaryME  ObjectAtom  ObjectAtomBipFE  ObjectAtomBipFENA  ObjectAtomBipME  AtomBinaryFE  AtomBinaryFENA  AtomBinaryME
#      │ String             Float64         Float64           Float64         Float64     Float64          Float64            Float64          Float64       Float64         Float64
# ─────┼──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
#    1 │ ipc23_ferry                   7.1               6.9             7.2         6.9              8.1                7.1              8.2          16.4            11.1          16.3
#    2 │ ipc23_rovers                 51.8              50.3            84.0        88.8             64.1               49.1             52.2         872.2           414.4         333.9
#    3 │ ipc23_blocksworld             9.4               7.9            10.2         7.2             11.0               10.1             10.3          23.5            17.4          21.8
#    4 │ ipc23_floortile              24.7              21.2            35.0        35.1             33.0               24.5             26.0         152.7            79.6          80.5
#    5 │ ipc23_satellite              17.4              17.2            27.5        26.5             24.4               19.0             21.4          93.4            52.5          61.8
#    6 │ ipc23_spanner                11.7               9.2            14.9        13.0             12.2               10.2             13.1          22.9            19.5          23.1
#    7 │ ipc23_childsnack              8.8              10.5            12.2        12.1             11.3                8.6             11.7          15.7            12.1          18.7
#    8 │ ipc23_miconic                33.8              25.6            34.2        39.5             43.7               30.8             34.7         327.0           147.8         136.4
#    9 │ ipc23_transport              14.5              10.6            19.4        22.3             17.9               12.9             13.0          75.5            40.7          46.4


   Row │ domain_name        ObjectBinaryFE  ObjectBinaryFENA  ObjectBinaryME  ObjectAtom  ObjectAtomBipFE  ObjectAtomBipFENA  ObjectAtomBipME  AtomBinaryFE  AtomBinaryFENA  AtomBinaryME
     │ String             Float64         Float64           Float64         Float64     Float64          Float64            Float64          Float64       Float64         Float64
─────┼──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
   1 │ ipc23_ferry                   4.9               3.4             4.2         6.6              6.5                4.2              4.9          15.2            12.5          15.6
   2 │ ipc23_rovers                 48.1              42.5            69.8        84.2             58.0               40.8             43.7        1705.7          1074.2         796.2
   3 │ ipc23_blocksworld             4.8               4.3             5.6         6.8              7.1                5.0              5.9          26.1            19.6          23.2
   4 │ ipc23_floortile              21.1              16.8            26.9        35.6             28.8               19.6             20.0         176.9            99.0         111.8
   5 │ ipc23_satellite              16.1              12.1            25.1        26.8             20.0               15.0             19.1         100.7            59.7          58.2
   6 │ ipc23_spanner                 9.3               5.8             8.7        13.9             10.4                6.3              6.8          26.1            21.0          25.1
   7 │ ipc23_childsnack              4.6               5.0             7.7        12.6              5.9                4.4              6.1          16.1            12.6          16.4
   8 │ ipc23_miconic                28.9              22.3            25.1        39.2             36.7               25.8             27.4         390.9           196.4         161.1
   9 │ ipc23_transport              11.7               9.7            16.2        22.6             15.4               10.0             11.6          86.2            46.6          52.5