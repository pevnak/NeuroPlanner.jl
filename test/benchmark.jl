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
using Accessors
using BenchmarkTools
using PrettyTables


highlight_max(data, i, j) = j > 1 && maximum(skipmissing(data[i,2:end])) == data[i,j]
highlight_min(data, i, j) = j > 1 && minimum(skipmissing(data[i,2:end])) == data[i,j]
nohighlight(args...) = false


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

function indistinguishability(archs, domain_name; difficulty="train")
	stat_type=:indistinguishability
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
		plan = load_plan(problem_file)
		ts = map(archs) do hnet
			pddld = hnet(domain; message_passes = graph_layers, residual)
			ds = LₛMiniBatch(pddld, domain, problem, plan)
			kb = ds.x
			# internal function deduplicate(kb::KnowledgeBase)
			ii = Int[]
			for k in keys(kb)
				new_entry, ii = NeuroPlanner._deduplicate(kb, kb[k])
				kb = replace(kb, k, new_entry)
			end
			sum(ii[i] == ii[j] for (i,j) in zip(ds.H₊.indices, ds.H₋.indices))
		end
		ns = tuple([Symbol("$(a)") for a in archs]...)
		stats = merge((;stat_type, domain_name, difficulty, problem_file), NamedTuple{ns}(ts))
		@show stats
		stats
	end
end

function format_value(xs::AbstractVector{<:AbstractString})
	format_value(map(JSON.parse,xs))
end

function format_value(xs::AbstractVector{<:Number})
	x = mean(xs)
	x isa Number && return(round(1e6*mean(x), digits = 1))
	"$(round(x[1], digits = 1)) / $(round(x[2], digits = 1))"
end

function format_value(xs::AbstractVector{<:Vector{<:Number}})
	x = mean(xs)
	"$(round(x[1], digits = 1)) / $(round(x[2], digits = 1))"
end

function fix_type(xs)
	T = typeof(first(xs))
	Vector{T}(map(T, xs))
end

function format_value(xs::AbstractVector{<:Any}, ::Val{ST}) where {ST}
	xx = collect(skipmissing(xs))
	isempty(xx) && return(missing)
	if first(xx) isa AbstractVector
		xx = fix_type(map(fix_type, xx))
	else
		xx = fix_type(xx)
	end
	format_value(xx)
end

function format_value(xs::AbstractVector{<:AbstractString}, st::Val{:vertices_and_edges}) where {ST}
	xx = map(xs) do x
		parse.(Float64, split(x[2:end-1], ","))
	end
	format_value(xx, st)
end

function format_value(xs::AbstractVector{<:Any}, ::Val{:indistinguishability})
	xx = collect(skipmissing(xs))
	isempty(xx) && return(missing)
	if first(xx) isa AbstractVector
		xx = fix_type(map(fix_type, xx))
	else
		xx = fix_type(xx)
	end
	round(mean(xx), digits = 2)
end

show_data(df::DataFrame, stat_type::Symbol) = show_data(df, Val(stat_type))

function show_data(df::DataFrame, stat_type::Val{ST}) where {ST}
	ST ∉ [:extractor, :extract_model, :vertices_and_edges, :indistinguishability] && error("stat type has to be in [:extractor, :extract_model, :vertices_and_edges]")
	fdf = filter(r -> Symbol(r.stat_type) == ST, df)
	gdf = DataFrames.groupby(fdf, ["domain_name"]);
	stat = combine(gdf) do sub_df 
		 (ObjectBinaryFE = format_value(sub_df.ObjectBinaryFE, stat_type),
		 	ObjectBinaryFENA = format_value(sub_df.ObjectBinaryFENA, stat_type),
		 	ObjectBinaryME = format_value(sub_df.ObjectBinaryME, stat_type),
		 	ObjectAtom = format_value(sub_df.ObjectAtom, stat_type), 
		 	ObjectAtomBipFE = format_value(sub_df.ObjectAtomBipFE, stat_type), 
		 	ObjectAtomBipFENA = format_value(sub_df.ObjectAtomBipFENA, stat_type), 
		 	ObjectAtomBipME = format_value(sub_df.ObjectAtomBipME, stat_type), 
		 	AtomBinaryFE = format_value(sub_df.AtomBinaryFE, stat_type), 
		 	AtomBinaryFENA = format_value(sub_df.AtomBinaryFENA, stat_type), 
		 	AtomBinaryME = format_value(sub_df.AtomBinaryME, stat_type), 
		 	ASNet = format_value(sub_df.ASNet, stat_type), 
		 	HGNN = format_value(sub_df.HGNN, stat_type), 
		 )
	end
	stat
end


function paper_table(df, highlighter = highlight_max)
	#add rank
	blank_col = fill(missing, size(df,1))
	problems = map(s -> replace(s, "ipc23_" => ""), df.domain_name)
	data = hcat(problems, blank_col,
			df.ObjectBinaryFE, df.ObjectBinaryFENA, df.ObjectBinaryME, blank_col, 
			df.AtomBinaryFE, df.AtomBinaryFENA, df.AtomBinaryME, blank_col,
			df.ObjectAtomBipFE, df.ObjectAtomBipFENA, df.ObjectAtomBipME, df.ObjectAtom, blank_col,
			 df.ASNet, df.HGNN)
	header = ["problem","",
		"ObjectBinaryFE", "ObjectBinaryfena", "ObjectBinaryme", "", 
		"AtomBinaryfe", "AtomBinaryfena", "AtomBinaryme", "",
			"ObjectAtomBipfe", "ObjectAtomBipfena", "ObjectAtomBipme", "ObjectAtom", "",
			 "ASNet", "HGNN"]

	h = [highlighter(data, i, j) for i in 1:size(data,1), j in 1:size(data,2)]
	h[h .=== missing] .= false
	# after highlighting remove missings
	data[:,[2,6,10,15]] .= ""

	# text backend
	highlighters = Highlighter((data, i, j) -> h[i, j], crayon"yellow")
	pretty_table(data ; header, backend = Val(:text) , highlighters)
	# latex backend
	highlighters = LatexHighlighter((data, i, j) -> h[i, j], ["textbf"])
	pretty_table(data ; header, backend = Val(:latex) , highlighters)
end



function run_and_tabulate()
	stat_types = [:extractor, :extract_model, :vertices_and_edges]
	archs = [ObjectBinaryFE, ObjectBinaryFENA, ObjectBinaryME, ObjectAtom, ObjectAtomBipFE, ObjectAtomBipFENA, ObjectAtomBipME, AtomBinaryFE, AtomBinaryFENA, AtomBinaryME, ASNet, HGNN]
	data = map(Iterators.product(IPC_PROBLEMS, stat_types)) do (problem, stat_type)
		benchmark_domain_arch(archs, problem; difficulty = "train", stat_type)
	end
	data = vec(data)
	df = DataFrame(reduce(vcat, data))

	# Table 2 in the main paper
	paper_table(show_data(df, :vertices_and_edges))

	# Table 3 in the main paper
	paper_table(show_data(df, :extract_model))

	# this one is not used
	# display(show_data(df, :extractor))

	data = map(IPC_PROBLEMS) do problem
		_archs = problem ∈ ("ipc23_miconic", "ipc23_sokoban") ? setdiff(archs,[ASNet, HGNN]) : archs
		indistinguishability(_archs, problem; difficulty = "train")
	end

	# Table 1 in supplementary material
	data = map(Base.Fix1(reduce, vcat), data)
	df = DataFrame()
	for shard in data
		for x in shard
			push!(df, x, cols = :union)
		end
	end
	df = DataFrame(reduce(vcat, vec(data)))
	paper_table(show_data(df, :indistinguishability))
end