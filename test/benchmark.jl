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

num_edges(ds::ArrayNode) = 0
num_edges(ds::AbstractMatrix) = 0
num_edges(ds::ProductNode) = mapreduce(num_edges, +, ds.data)
num_edges(ds::BagNode) = mapreduce(length, +, ds.bags) + num_edges(ds.data)

function graph_stats(kb::KnowledgeBase)
	if haskey(kb.kb, :x1)
		nv = size(kb[:x1],2)
		ne = num_edges(kb[:gnn_2])
		return([nv, ne])
	else # this is for ASNet and HGNN
		nv = numobs(kb[:pred_1])
		ks = collect(setdiff(keys(kb.kb), [:o, :pred_1]))
		ne = mapreduce(k -> num_edges(kb[k]), +, ks)
		return([nv,ne])
	end
end


function time_extraction(pddle, model, states)
	map(pddle, states)
	mean(@elapsed map(pddle, states) for _ in 1:100) / length(states)
end

compute_stats(::Val{:extractor}, args...) = time_extraction(args...)

function time_model_extraction(pddle, model, states)
	map(model ∘ pddle, states)
	mean(@elapsed map(model ∘ pddle, states) for _ in 1:100) / length(states)
end

compute_stats(::Val{:extract_model}, args...) = time_model_extraction(args...)

function time_model_dedu_extraction(pddle, model, states)
	map(model ∘ deduplicate ∘ pddle, states)
	mean(@elapsed map(model ∘ deduplicate ∘ pddle, states) for _ in 1:100) / length(states)
end

compute_stats(::Val{:extract_dedu_model}, args...) = time_model_dedu_extraction(args...)

function vertices_and_edges(pddle, model, states)
	mean(map(graph_stats ∘ pddle, states))
end

compute_stats(::Val{:vertices_and_edges}, args...) = vertices_and_edges(args...)

function benchmark_domain_arch(archs, domain_name; difficulty="train", stat_type=:extractor)
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

		ts = map(archs, models) do hnet, model
			pddld = hnet(domain; message_passes = graph_layers, residual)
			pddle, state = initproblem(pddld, problem)
			compute_stats(Val(stat_type), pddle, model, states)
		end
		ns = tuple([Symbol("$(a)") for a in archs]...)
		stats = merge((;stat_type, domain_name, difficulty, problem_file), NamedTuple{ns}(ts))
		@show stats
		stats
	end
end

function format_value(xs::AbstractVector)
	x = mean(xs)
	x isa Number && return(round(1e6*mean(x), digits = 1))
	"$(round(x[1], digits = 1)) / $(round(x[2], digits = 1))"
end

function show_data(df::DataFrame, stat_type)
	stat_type ∉ [:extractor, :extract_model, :vertices_and_edges] && error("stat type has to be in [:extractor, :extract_model, :vertices_and_edges]")
	fdf = filter(r -> Symbol(r.stat_type) == stat_type, df)
	gdf = DataFrames.groupby(fdf, ["domain_name"]);
	stat = combine(gdf) do sub_df 
		 (ObjectBinaryFE = format_value(sub_df.ObjectBinaryFE),
		 	ObjectBinaryFENA = format_value(sub_df.ObjectBinaryFENA),
		 	ObjectBinaryME = format_value(sub_df.ObjectBinaryME),
		 	ObjectAtom = format_value(sub_df.ObjectAtom), 
		 	ObjectAtomBipFE = format_value(sub_df.ObjectAtomBipFE), 
		 	ObjectAtomBipFENA = format_value(sub_df.ObjectAtomBipFENA), 
		 	ObjectAtomBipME = format_value(sub_df.ObjectAtomBipME), 
		 	AtomBinaryFE = format_value(sub_df.AtomBinaryFE), 
		 	AtomBinaryFENA = format_value(sub_df.AtomBinaryFENA), 
		 	AtomBinaryME = format_value(sub_df.AtomBinaryME), 
		 	ASNet = format_value(sub_df.ASNet), 
		 	HGNN = format_value(sub_df.HGNN), 
		 )
	end
	stat
end


function all_benchmarks()
	stat_types = [:extractor, :extract_model, :vertices_and_edges]
	archs = [ObjectBinaryFE, ObjectBinaryFENA, ObjectBinaryME, ObjectAtom, ObjectAtomBipFE, ObjectAtomBipFENA, ObjectAtomBipME, AtomBinaryFE, AtomBinaryFENA, AtomBinaryME, ASNet, HGNN]
	data = map(Iterators.product(setdiff(IPC_PROBLEMS,["ipc23_sokoban"]), stat_types)) do (problem, stat_type)
		benchmark_domain_arch(archs, problem; difficulty = "train", stat_type)
	end
	data = vec(data)
	df = DataFrame(reduce(vcat, data))
	githash = readchomp(`git rev-parse --verify HEAD`)
	CSV.write("benchmarks/stats_$(githash).csv", df)
	display(show_data(df, :extractor))
	display(show_data(df, :extract_model))
	display(show_data(df, :vertices_and_edges))
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

# GNN edges without central vertex
    Row │ domain_name        ObjectBinaryFE  ObjectBinaryFENA  ObjectBinaryME  ObjectAtom  ObjectAtomBipFE  ObjectAtomBipFENA  ObjectAtomBipME  AtomBinaryFE  AtomBinaryFENA  AtomBinaryME  ASNet    HGNN
     │ String             Float64         Float64           Float64         Float64     Float64          Float64            Float64          Float64       Float64         Float64       Float64  Float64
─────┼────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
   1 │ ipc23_ferry                  21.7              22.6            22.5        27.6             41.6               39.0             41.7          48.2            51.2          53.1    234.2    262.3
   2 │ ipc23_rovers                139.0             366.3           419.2       418.2            480.6              448.3            403.5        8565.1          8887.3        5047.9  15328.3  15321.6
   3 │ ipc23_blocksworld            23.4              23.3            23.7        28.1             42.2               39.9             41.3         106.5           110.0         104.2    590.2    563.5
   4 │ ipc23_floortile              86.9              86.0           169.6       197.3            184.2              173.1            151.7        1276.9          1271.3         747.0  13377.8  13644.4
   5 │ ipc23_satellite              74.1              71.0           140.4       160.1            141.1              130.0            121.7         736.8           717.3         417.9   5615.9   5822.1
   6 │ ipc23_spanner                30.2              28.1            63.8        70.0             48.8               46.6             47.0          94.6            94.1         104.9    631.1    659.8
   7 │ ipc23_childsnack             18.4              19.0            55.5        63.9             26.7               26.1             31.1          53.6            46.8          59.4    429.5    469.9
   8 │ ipc23_miconic               117.1             101.1            99.2       132.3            229.4              215.6            191.9        2941.4          2781.0        1500.1   1199.5   1268.1
   9 │ ipc23_transport              50.4              48.2            94.4       117.2             96.2               90.4             83.9         646.6           632.5         333.0   4113.8   3911.1

# GNN edges with central vertex
 Row │ domain_name        ObjectBinaryFE  ObjectBinaryFENA  ObjectBinaryME  ObjectAtom  ObjectAtomBipFE  ObjectAtomBipFENA  ObjectAtomBipME  AtomBinaryFE  AtomBinaryFENA  AtomBinaryME  ASNet    HGNN
     │ String             Float64         Float64           Float64         Float64     Float64          Float64            Float64          Float64       Float64         Float64       Float64  Float64
─────┼────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
   1 │ ipc23_ferry                  26.2              29.0            30.1        30.9             51.3               46.6             51.2          55.5            58.0          61.7    245.1    272.1
   2 │ ipc23_rovers                149.6             367.1           493.2       441.3            527.9              501.4            451.0        8410.0          8796.6        4957.7  14900.7  15114.7
   3 │ ipc23_blocksworld            25.4              29.3            26.1        28.9             51.8               48.6             46.4         115.5           119.2         110.8    579.5    559.0
   4 │ ipc23_floortile              93.4              93.1           177.2       203.2            202.3              194.3            171.4        1254.8          1243.4         738.6  12819.3  13089.3
   5 │ ipc23_satellite              81.0              73.0           155.6       171.6            168.1              175.9            135.1         784.1           708.1         417.4   5451.5   5631.4
   6 │ ipc23_spanner                34.9              32.8            70.0        71.5             60.6               53.6             64.3         104.1           103.8         114.8    629.3    661.9
   7 │ ipc23_childsnack             25.7              25.6            62.8        60.3             35.7               35.9             35.8          59.8            58.0          64.2    444.3    481.0
   8 │ ipc23_miconic               119.7             112.9           104.1       137.9            264.8              243.5            228.9        2935.4          2786.4        1469.0   1175.7   1285.7
   9 │ ipc23_transport              60.1              49.2            97.8       118.5            120.0              113.2             92.0         659.8           632.2         332.4   4038.6   3868.3

 Row │ domain_name        ObjectBinaryFE  ObjectBinaryFENA  ObjectBinaryME  ObjectAtom  ObjectAtomBipFE  ObjectAtomBipFENA  ObjectAtomBipME  AtomBinaryFE  AtomBinaryFENA  AtomBinaryME  ASNet    HGNN
     │ String             Float64         Float64           Float64         Float64     Float64          Float64            Float64          Float64       Float64         Float64       Float64  Float64
─────┼────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
   1 │ ipc23_ferry                  28.3              30.9            30.8        34.4             53.6               52.2             49.9          59.8            63.0          67.5    263.3    293.4
   2 │ ipc23_rovers                146.2             356.3           500.9       439.8            539.7              489.3            446.3        7693.6          8341.3        4985.8  14910.4  14975.8
   3 │ ipc23_blocksworld            28.7              26.0            26.2        31.2             51.0               48.6             50.0         114.6           120.0         113.8    587.7    555.2
   4 │ ipc23_floortile              92.2              95.2           177.9       213.9            214.7              192.3            177.5        1186.7          1225.4         726.2  12935.7  13159.6
   5 │ ipc23_satellite              70.8              75.4           136.1       186.0            160.0              168.6            127.1         711.6           701.8         415.8   5492.2   5799.3
   6 │ ipc23_spanner                31.1              34.5            75.2        63.0             69.7               46.9             61.9          99.2           104.8         113.1    633.9    671.5
   7 │ ipc23_childsnack             27.8              28.0            60.1        58.4             44.1               36.1             44.9          49.8            63.1          70.7    444.2    485.5
   8 │ ipc23_miconic               122.0             108.0           102.1       135.1            276.2              245.5            215.7        2773.4          2742.9        1518.0   1203.9   1310.6
   9 │ ipc23_transport              50.0              50.9            75.7        98.9            105.8               96.2             83.2         659.3           705.1         339.3   4313.8   4006.9



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

    Row │ domain_name        ObjectBinaryFE  ObjectBinaryFENA  ObjectBinaryME  ObjectAtom  ObjectAtomBipFE  ObjectAtomBipFENA  ObjectAtomBipME  AtomBinaryFE  AtomBinaryFENA  AtomBinaryME
     │ String             Float64         Float64           Float64         Float64     Float64          Float64            Float64          Float64       Float64         Float64
─────┼──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
   1 │ ipc23_ferry                   5.1               3.8             4.2         6.4              7.1                5.1              4.9          16.4            13.9          15.4
   2 │ ipc23_rovers                 50.4              47.7            57.1        81.6             62.8               47.8             43.4        2122.9          1455.8         599.2
   3 │ ipc23_blocksworld             6.0               4.5             4.9         6.5              7.8                5.4              5.4          27.2            20.6          21.5
   4 │ ipc23_floortile              23.2              18.6            22.3        34.8             30.6               21.9             19.4         185.9           124.6          75.6
   5 │ ipc23_satellite              19.2              14.0            20.4        27.1             24.5               17.4             16.4         118.8            84.2          65.6
   6 │ ipc23_spanner                 7.2               5.8             9.7        12.7             10.4                6.7              8.5          26.2            24.2          24.1
   7 │ ipc23_childsnack              4.7               4.2             5.7        12.3              6.7                4.6              5.5          16.2            14.7          16.1
   8 │ ipc23_miconic                32.6              24.3            23.9        39.6             42.7               31.1             27.5         467.3           277.2         160.4
   9 │ ipc23_transport              14.1               9.7            15.0        21.7             17.4               11.5             11.6          99.6            63.5          45.2
