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

function add_goalstate(pddle::PDDLExtractor{<:Nothing,<:Nothing}, problem)
	goal =  goalstate(pddle.domain, problem)
	term2id = Dict(only(get_args(v)) => i for (i, v) in enumerate(goal.types))
	goal = multigraph(pddle, goal, term2id)
	PDDLExtractor(pddle.domain, pddle.binary_predicates, pddle.nunanary_predicates, term2id, goal)
end

function initproblem(pddle::PDDLExtractor{<:Nothing,<:Nothing}, problem)
	add_goalstate(pddle, problem), initstate(pddle.domain, problem)
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
struct MultiGraph{N,G,T<:AbstractMatrix,C}
	graphs::NTuple{N,G}
	vprops::T
	components::C
end

SparseMultiGraph = MultiGraph{<:Any,<:SparseGraph,<:Any,<:Any}
SimpleMultiGraph = MultiGraph{<:Any,<:SimpleGraph,<:Any,<:Any}

MultiGraph(graphs::Vector, args...) = MultiGraph(tuple(graphs...), args...)

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
	MultiGraph(graphs, vprops, nothing)
end

"""
function Base.vcat(a::MultiGraph, b::MultiGraph)
	
concatenate graphs and their features. It is assumed
the vertices are aligned
"""
function Base.vcat(a::MultiGraph, b::MultiGraph)
	a.components != b.components && error("components has to be equal when concatenating graphs")
	all(nv(a.graphs[1]) == nv(g) for g in a.graphs)
	all(nv(a.graphs[1]) == nv(g) for g in b.graphs)
	vprops = vcat(a.vprops, b.vprops)
	MultiGraph((a.graphs..., b.graphs...), vprops, a.components)
end

function Base.reduce(::typeof(Base.cat), mgs::Vector{<:SimpleMultiGraph})
	# TODO: consider to add some sane asserts
	components = map(g -> g.components === nothing ? [1:size(g.vprops,2)] : g.components, mgs)
	fadj = [map(g -> g.graphs[i].fadjlist, mgs) for i in 1:length(first(mgs).graphs)]
	offset = 0
	for (i, g) in enumerate(mgs)
		components[i] = map(c -> c .+ offset, components[i])
		for j in 1:length(fadj)
			fadj[j][i] = map(c -> c .+ offset, fadj[j][i])
		end
		offset += size(g.vprops, 2)
	end
	components = reduce(vcat, components)
	fadj = map(fa -> reduce(vcat, fa), fadj)
	graphs = tuple(map(fa -> SimpleGraph(sum(length.(fa)), fa), fadj)...)
	vprops = reduce(hcat, [g.vprops for g in mgs])
	MultiGraph(graphs, vprops, components)
end

function sparsegraph(g::SimpleMultiGraph) 
	MultiGraph(map(g -> FeaturedGraph(g).graph, g.graphs), g.vprops, g.components)
end




using GraphSignals: EdgeSignal, GlobalSignal, NodeDomain

"""
struct MultiGNNLayer{G<:Tuple}
	gconvs::G
end

implements one graph convolution layer over the MultiGraph. 
convolutions stored in `gconvs` are assumed to come from 
`GeometricFlux`, hence the `MultiGNNLayer` can process only 
`MultiGraph{<:NTuple{N, <:SparseGraph}`, where 
`SparseGraph` is a type from `GraphSignals.jl.` 
If `MultiGraph{<:NTuple{N, <:SimpleGraph}` is supplied, it 
internally converted to `MultiGraph{<:NTuple{N, <:SparseGraph}`
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

function (mg::MultiGNNLayer)(g::SparseMultiGraph)
	vprops = map(mg.gconvs, g.graphs) do gconv, h 
		gconv(_construct_fg(h, g.vprops)).nf.signal
	end 
	vprops = vcat(vprops...)
	MultiGraph(g.graphs, vprops, g.components)
end

function (mg::MultiGNNLayer)(g::SimpleMultiGraph)
	mg(sparsegraph(g))
end

_construct_fg(g::SparseGraph, nf) = FeaturedGraph(g, nf,  EdgeSignal(nothing), GlobalSignal(nothing), NodeDomain(nothing), :adjm)


"""
meanmax(g::MultiGraph)

Reduction function over the features of MultiGraph. Understands components
"""
function meanmax(g::MultiGraph{<:Any,<:Any,<:Any,<:Nothing})
	vprops = g.vprops
	vcat(mean(vprops, dims = 2), maximum(vprops, dims = 2))
end

function meanmax(g::MultiGraph{<:Any,<:Any,<:Any,<:Vector{<:UnitRange}})
	vprops = g.vprops
	xx = map(g.components) do c 
		x = (@view vprops[:,c])
		vcat(mean(x, dims = 2), maximum(x, dims = 2))
	end
	reduce(hcat, xx)
end


struct MultiModel{G<:Tuple,D}
	g::G
	d::D
end 

Flux.@functor MultiModel

function MultiModel(h₀::MultiGraph, odim::Int, maked)
	m₁ = PDDL2Graph.MultiGNNLayer(h₀, odim)
	h₁ = m₁(h₀)
	m₂ = PDDL2Graph.MultiGNNLayer(h₁, odim)
	h₂ = m₂(h₁)
	h = vcat(PDDL2Graph.meanmax(h₁), PDDL2Graph.meanmax(h₂))
	d = maked(size(h,1))
	MultiModel((m₁,m₂), d)
end

function (mm::MultiModel)(h₀::MultiGraph)
	h₁ = mm.g[1](h₀)
	h₂ = mm.g[2](h₁)
	h = vcat(PDDL2Graph.meanmax(h₁), PDDL2Graph.meanmax(h₂))
	mm.d(h)
end

function _multimodel(layers, h)
	isempty(layers) && return(h)
	_multimodel(Base.tail(layers), first(layers)(h))
end