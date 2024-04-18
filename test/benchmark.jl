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

function benchmark_domain_arch(archs, domain_name)
	graph_layers = 2
	dense_dim = 16
	dense_layers = 2
	residual = "none"

	residual = Symbol(residual)
	domain_pddl, problem_files = getproblem(domain_name, false)
	train_files = filter(s -> isfile(plan_file(s)), problem_files)
	train_files = domain_name ∉ IPC_PROBLEMS ? sample(train_files, min(div(length(problem_files), 2), length(train_files)), replace = false) : train_files
	domain = load_domain(domain_pddl)

	# function experiment(domain_name, hnet, domain_pddl, train_files, problem_files, filename, fminibatch;max_steps = 10000, max_time = 30, graph_layers = 2, residual = true, dense_layers = 2, dense_dim = 32, settings = nothing)
	models = map(archs) do hnet
		pddld = hnet(domain; message_passes = graph_layers, residual)
		problem = load_problem(first(train_files))
		pddle, state = initproblem(pddld, problem)
		h₀ = pddle(state)
		reflectinmodel(h₀, d -> Dense(d, dense_dim, relu);fsm = Dict("" =>  d -> ffnn(d, dense_dim, 1, dense_layers)))
	end

	map(train_files[end-5:end]) do problem_file
		problem = load_problem(problem_file)
		plan = load_plan(problem_file)
		state = initstate(domain, problem)
		states = SymbolicPlanners.simulate(StateRecorder(), domain, state, plan)

		# ts = map(archs) do hnet
		# 	pddld = hnet(domain; message_passes = graph_layers, residual)
		# 	pddle, state = initproblem(pddld, problem)
		# 	map(pddle, states)
		# 	mean(@elapsed map(pddle, states) for _ in 1:100) / length(states)
		# end

		ts = map(models, archs) do model, hnet
			pddld = hnet(domain; message_passes = graph_layers, residual)
			pddle, state = initproblem(pddld, problem)
			map(model ∘ pddle, states)
			mean(@elapsed map(model ∘ pddle, states) for _ in 1:100) / length(states)
		end
		ns = tuple([Symbol("$(a)") for a in archs]...)
		stats = merge((;domain_name, problem_file), NamedTuple{ns}(ts))
		@show stats
		stats
	end
end

archs = [ObjectBinary, MixedLRNN2, ObjectAtom, AtomBinary, ObjectPair]
data = map(problem -> benchmark_domain_arch(archs, problem), setdiff(IPC_PROBLEMS,["ipc23_sokoban"]))
df = DataFrame(reduce(vcat, data))
gdf = DataFrames.groupby(df, ["domain_name"])
combine(gdf) do sub_df 
	 (;ObjectBinary = round(1e6*mean(sub_df.ObjectBinary), digits = 1),
	 	MixedLRNN2 = round(1e6*mean(sub_df.MixedLRNN2), digits = 1), 
	 	ObjectAtom = round(1e6*mean(sub_df.ObjectAtom), digits = 1), 
	 	AtomBinary = round(1e6*mean(sub_df.AtomBinary), digits = 1),
	 	ObjectPair = round(1e6*mean(sub_df.ObjectPair), digits = 1),
	 )
end

# f0a267e17a8bce06bd0f88233902279471f6f605
# EdgeBuilderComp on M3 in μs with a HMGNNnetwork with config (graph_layers = 2, dense_dim = 16, dense_layers = 2, residual = "none")
#  Row │ domain_name        ObjectBinary  MixedLRNN2  ObjectAtom  AtomBinary  ObjectPair
#      │ String             Float64       Float64     Float64     Float64     Float64
# ─────┼─────────────────────────────────────────────────────────────────────────────────
#    1 │ ipc23_ferry                44.1        33.3        31.6       141.8  16597.2
#    2 │ ipc23_rovers              582.8       596.8       501.4     55014.8      1.04e5
#    3 │ ipc23_blocksworld          53.5        35.7        34.4       360.4   6713.5
#    4 │ ipc23_floortile           257.1       276.2       240.8      5212.1  45793.2
#    5 │ ipc23_satellite           215.5       241.5       212.8      2280.9  74582.5
#    6 │ ipc23_spanner             111.9        96.9        86.1       336.2  20747.5
#    7 │ ipc23_childsnack           92.1        85.0        78.0       268.0  21443.0
#    8 │ ipc23_miconic             175.3       199.8       162.0     16673.5  31035.6
#    9 │ ipc23_transport           158.5       164.3       145.0      1322.4  18914.3



# f0a267e17a8bce06bd0f88233902279471f6f605
# EdgeBuilderComp on M3 in μs
#  Row │ domain_name        ObjectBinary  MixedLRNN2  ObjectAtom  AtomBinary  ObjectPair
#      │ String             Float64       Float64     Float64     Float64     Float64
# ─────┼─────────────────────────────────────────────────────────────────────────────────
#    1 │ ipc23_ferry                 9.7         9.7         9.2        45.1       305.8
#    2 │ ipc23_rovers              107.8       252.6       114.6      5197.7     11985.7
#    3 │ ipc23_blocksworld          13.7         9.6         8.8        82.9       133.2
#    4 │ ipc23_floortile            55.0        84.7        45.9       822.0      4835.6
#    5 │ ipc23_satellite            39.5        66.7        39.1       502.0      7333.0
#    6 │ ipc23_spanner              22.7        20.6        17.1        90.2       820.3
#    7 │ ipc23_childsnack           19.9        14.5        13.6        75.9       352.8
#    8 │ ipc23_miconic              49.7        84.3        49.4      1680.4      5590.2
#    9 │ ipc23_sokoban             119.8       105.0        76.4     10670.8       ---
#   10 │ ipc23_transport            28.8        46.9        27.3       219.0      1216.3


# f0a267e17a8bce06bd0f88233902279471f6f605
# EdgeBuilderCompMat on M3 in μs
#  Row │ domain_name        ObjectBinary  MixedLRNN2  ObjectAtom  AtomBinary
#      │ String             Float64       Float64     Float64     Float64
# ─────┼─────────────────────────────────────────────────────────────────────
#    1 │ ipc23_ferry                10.6         9.4         8.6        53.8
#    2 │ ipc23_rovers              108.7       270.8       116.8      4812.0
#    3 │ ipc23_blocksworld          14.5         9.8         9.7        89.8
#    4 │ ipc23_floortile            52.7        84.8        45.8       806.6
#    5 │ ipc23_satellite            38.6        54.8        34.3       496.4
#    6 │ ipc23_spanner              24.4        22.4        17.1        93.3
#    7 │ ipc23_childsnack           20.2        14.8        14.3        74.9
#    8 │ ipc23_miconic              50.4        83.7        46.3      1679.7
#    9 │ ipc23_sokoban             116.3       109.6        75.7     10611.4
#   10 │ ipc23_transport            28.7        46.0        29.6       213.0

# [vitek ce6a7bf] ScatteredBags on M3 in μs
#  Row │ domain_name        ObjectBinary  MixedLRNN2  ObjectAtom  AtomBinary
#      │ String             Float64       Float64     Float64     Float64
# ─────┼─────────────────────────────────────────────────────────────────────
#    1 │ ipc23_ferry                18.0        14.2        13.0        94.4
#    2 │ ipc23_rovers              207.8       335.2       196.5      8110.4
#    3 │ ipc23_blocksworld          22.9        14.0        13.1       179.2
#    4 │ ipc23_floortile            96.6       122.4        85.6      1344.5
#    5 │ ipc23_satellite            72.8        83.5        72.0       770.5
#    6 │ ipc23_spanner              38.2        35.1        30.6       190.8
#    7 │ ipc23_childsnack           33.4        27.1        25.9       154.2
#    8 │ ipc23_miconic              81.3       112.4        73.1      2725.0
#    9 │ ipc23_sokoban             223.3       176.4       135.9     12285.1
#   10 │ ipc23_transport            54.7        68.8        48.9       445.7
