"""
	PDDLExtractor{D,G}

	contains information about converting problem described in PDDL 
	to graph representation.
"""
struct PDDLExtractor{D,G,DO}
	domain::DO
	binary_predicates::Dict{Symbol,Int64}
	nunanary_predicates::Dict{Symbol,Int64}
	term2id::D
	goal::G
end

function PDDLExtractor(domain, problem; embed_goal = true)
	dictmap(x) = Dict(reverse.(enumerate(sort(x))))
	predicates = collect(domain.predicates)
	any(kv -> length(kv[2].args) > 2, predicates) && error("Cannot interpret domains with more than binary predicates")
	binary_predicates = dictmap([kv[1] for kv in predicates if length(kv[2].args) == 2])
	nunanary_predicates = dictmap([kv[1] for kv in predicates if length(kv[2].args) ≤  1])
	pddle = PDDLExtractor(domain, binary_predicates, nunanary_predicates, nothing, nothing)
	!embed_goal && return(pddle)
	add_goalstate(pddle, problem)
end

function PDDLExtractor(domain)
	dictmap(x) = Dict(reverse.(enumerate(sort(x))))
	predicates = collect(domain.predicates)
	any(kv -> length(kv[2].args) > 2, predicates) && error("Cannot interpret domains with more than binary predicates")
	binary_predicates = dictmap([kv[1] for kv in predicates if length(kv[2].args) == 2])
	nunanary_predicates = dictmap([kv[1] for kv in predicates if length(kv[2].args) ≤  1])
	PDDLExtractor(domain, binary_predicates, nunanary_predicates, nothing, nothing)
end

"""
add_goalstate(pddle, problem)
add_goalstate(pddle, problem, goal)

adds goal state to the extract, such that the goal descripto is always add 
to the graph.
"""
function add_goalstate(pddle::PDDLExtractor{<:Nothing,<:Nothing}, problem)
	add_goalstate(pddle, problem, goalstate(pddle.domain, problem))
end

function add_goalstate(pddle::PDDLExtractor{<:Nothing,<:Nothing}, problem, goal)
	pddle = specialize(pddle, problem)
	goal = multigraph(pddle, goal)
	PDDLExtractor(pddle.domain, pddle.binary_predicates, pddle.nunanary_predicates, pddle.term2id, goal)
end

function add_goalstate(pddle::PDDLExtractor{<:Any,<:Nothing}, problem, goal)
	goal = multigraph(pddle, goal)
	PDDLExtractor(pddle.domain, pddle.binary_predicates, pddle.nunanary_predicates, pddle.term2id, goal)
end

"""
specialize(pddle::PDDLExtractor{<:Nothing,<:Nothing}, problem)

initializes extractor for a given `problem` by initializing mapping 
from objects to id of vertices. Goals are not changed added to the 
extractor.
"""
function specialize(pddle::PDDLExtractor{<:Nothing,<:Nothing}, problem)
	term2id = Dict(v => i for (i, v) in enumerate(problem.objects))
	PDDLExtractor(pddle.domain, pddle.binary_predicates, pddle.nunanary_predicates, term2id, nothing)
end

"""
initproblem(pddld::PDDLExtractor{<:Nothing,<:Nothing}, problem; add_goal = true)

Specialize extractor for the given problem instance and return init state 
"""
function initproblem(pddld::PDDLExtractor{<:Nothing,<:Nothing}, problem; add_goal = true)
	pddle = add_goal ? add_goalstate(pddld, problem) : specialize(pddld, problem)
	pddle, initstate(pddld.domain, problem)
end

"""
struct MultiGraph{G, T}
	graphs::G
	vprops::T
end

A simple container for multi-graph, where edges are of different types.
The idea is that `graphs` is a tuple of `SimpleGraph`, where one graph 
defines edges of type. `vprops` contains properties of vertices, which are shared
among graphs
"""
struct MultiGraph{N,G<:GNNGraph,T<:AbstractMatrix}
	graphs::NTuple{N,G}
	vprops::T

	function MultiGraph(gs::NTuple{N,G}, x::T) where {N,G,T}
		N < 1 && error("multigraph has to contain at least one graph, needed to identify components. Use empty Graph")
		new{N,G,T}(gs, x)
	end
end

MultiGraph(graphs::Vector, vprops) = MultiGraph(tuple(graphs...), vprops)
MultiGraph(graphs::NTuple{<:Any,<:SimpleGraph}, vprops) = MultiGraph(map(GNNGraph, graphs), vprops)

"""
	multigraph(pddle::PDDLExtractor{<:Dict,<:MultiGraph}, state)
	multigraph(pddle, state, term2id)

	convert state from pddl to MultiGraph. If pddle contains a GoalState, it is 
	automatically added to the multigraph
"""
function multigraph(pddle::PDDLExtractor{<:Dict,<:MultiGraph}, state)
	vcat(multigraph(pddle, state, pddle.term2id), pddle.goal)
end

(pddle::PDDLExtractor{<:Dict,<:MultiGraph})(state) = vcat(multigraph(pddle, state, pddle.term2id), pddle.goal)

function multigraph(pddle::PDDLExtractor{<:Dict,<:Nothing}, state)
	multigraph(pddle, state, pddle.term2id)
end

function multigraph(pddle::PDDLExtractor, state, term2id)
	vprops = zeros(Float32, length(pddle.nunanary_predicates), length(term2id))
	graphs = [Graph(length(term2id)) for _ in 1:length(pddle.binary_predicates)]
	for f in state.facts
		a = get_args(f)
		if length(a) == 2
			g = graphs[pddle.binary_predicates[f.name]]
			i, j = term2id[a[1]],  term2id[a[2]]
			e = (i < j) ? Edge(i, j) : Edge(j, i)
			add_edge!(g, e)
		elseif length(a) == 1 
			pid = pddle.nunanary_predicates[f.name]
			vid = term2id[only(get_args(f))]
			vprops[pid, vid] = 1
		else
			pid = pddle.nunanary_predicates[f.name]
			vprops[pid,:] .= 1
		end
	end
	foreach(g -> foreach(i -> add_edge!(g, Edge(i,i)), 1:nv(g)), graphs)
	MultiGraph(graphs, vprops)
end

"""
function Base.vcat(a::MultiGraph, b::MultiGraph)
	
concatenate graphs and their features. It is assumed
the vertices are aligned
"""
function Base.vcat(a::MultiGraph, b::MultiGraph)
	all(nv(a.graphs[1]) == nv(g) for g in a.graphs)
	all(nv(a.graphs[1]) == nv(g) for g in b.graphs)
	vprops = vcat(a.vprops, b.vprops)
	MultiGraph((a.graphs..., b.graphs...), vprops)
end

function GraphNeuralNetworks.batch(gs::Vector{<:MultiGraph})
	graphs = [batch([g.graphs[i] for g in gs]) for i in 1:length(gs[1].graphs)]
	graphs = tuple(graphs...)
	x = reduce(hcat, [g.vprops for g in gs])
	MultiGraph(graphs, x)
end

"""
struct MultiGNNLayer{G<:Tuple}
	gconvs::G
end

implements one graph convolution layer over the MultiGraph. 
convolutions stored in `gconvs` are assumed to come from 
`GeometricFlux`, hence the `MultiGNNLayer` can process only 
`MultiGraph{<:NTuple{N, <:GNNGraph}`, where 
`GNNGraph` is a type from `GraphSignals.jl.` 
If `MultiGraph{<:NTuple{N, <:SimpleGraph}` is supplied, it 
internally converted to `MultiGraph{<:NTuple{N, <:GNNGraph}`
and forwared for further processing.
"""
struct MultiGNNLayer{G<:Tuple}
	gconvs::G
end

MultiGNNLayer(gconvs::Vector) = MultiGNNLayer(tuple(gconvs...))

Flux.@functor MultiGNNLayer

function MultiGNNLayer(a::MultiGraph, odim)
	gconvs = map(a.graphs) do g
		idim = size(a.vprops, 1)
		GATConv(idim => odim, relu)
	end 
	MultiGNNLayer(tuple(gconvs...))
end

function (mg::MultiGNNLayer)(g::MultiGraph)
	vprops = map(mg.gconvs, g.graphs) do gconv, h 
		gconv(set_vertex_properties(h, g.vprops)).ndata.x
	end 
	vprops = vcat(vprops...)
	MultiGraph(g.graphs, vprops)
end

set_vertex_properties(g::GNNGraph, nf) = GNNGraph(g; ndata = nf)

struct MultiModel{G<:Tuple,D}
	g::G
	d::D
end 

Flux.@functor MultiModel

"""
MultiModel(h₀::MultiGraph, odim::Int, nlayers, maked)

construct GNN for MultiModel with `nlayers` of GATs, each with output dimension
`odim` and `nlayers`. The dense part after the aggregation will be constructed 
by `maked(d),` where `d` is the input dimension 
"""
function MultiModel(h₀::MultiGraph, odim::Int, nlayers, maked)
	layers = tuple()
	hs = (h₀,)
	h = h₀
	for i in 1:nlayers
		m = NeuroPlanner.MultiGNNLayer(h, odim)
		h = m(h)
		hs = (hs..., h)
		layers = (layers..., m)
	end
	h = vcat(map(meanmax, hs)...)
	d = maked(size(h,1))
	MultiModel(layers, d)
end

function apply_glayers(layers, h)
	isempty(layers) && return((h,))
	return(h, apply_glayers(layers[2:end], layers[1](h))...)
end

function (mm::MultiModel)(h₀::MultiGraph)
	hs = apply_glayers(mm.g, h₀)
	h = vcat(map(meanmax, hs)...)
	mm.d(h)
end

function meanmax(h::MultiGraph)
	g = h.graphs[1]
	x = h.vprops
	vcat(reduce_nodes(mean, g, x),reduce_nodes(max, g, x))
end