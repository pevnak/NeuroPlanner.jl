module NeuroPlanner

using PDDL
using Julog
using Graphs
using GraphSignals
using Flux
using GeometricFlux
using Statistics
using SymbolicPlanners

include("multigraph.jl")
export PDDLExtractor, MultiGraph, MultiGNN, FeaturedMultiGraph, MultiModel, initproblem

"""
	function Graphs.Graph(state::GenericState)
	
	Create a graph describing relations in pddl.
	add a goal state `from problem.goal`
"""
function state2graph(state::GenericState)
	term2id = Dict(only(get_args(v)) => i for (i, v) in enumerate(state.types))
	_construct_graph(term2id, state.facts)
end


"""
	function goal2graph(term2id, problem)
	function goal2graph(term2id, facts)

	construct a graph according to facts, where `term2id` is a name of 
	term to its id 
"""
function goal2graph(term2id, problem::GenericProblem)
	goal2graph(term2id, problem.goal)
end

function goal2graph(term2id, facts::Compound)
	facts.name != :and && error("can parse ony single goal, not $(facts.name)")
	goal2graph(term2id, get_args(facts))
end

function goal2graph(term2id, facts::Vector{Term})
	_construct_graph(term2id, facts)
end

function _construct_graph(term2id, facts)
	g = Graph(length(term2id))
	_construct_graph!(g, term2id, facts)
end

function _construct_graph!(g::AbstractGraph, term2id, facts)
	vprops = Dict{Int,Any}()
	eprops = Dict{Edge,Any}()
	for f in facts
		s = f.name
		a = get_args(f)
		if length(a) == 1 
			k = term2id[only(a)]
			vprops[k] = push!(get(vprops, k, []), s)
		elseif length(a) == 2
			i, j = term2id[a[1]],  term2id[a[2]]
			e = (i < j) ? Edge(i, j) : Edge(j, i)
			add_edge!(g, e)
			eprops[e] = push!(get(eprops, e, []), s)
		else 
			@error "ignoring $(s) with $(a)"
		end
	end
	(;graph = g, vprops, eprops, term2id)
end

function _construct_featured_graph(term2id, facts)
	(g, vprops, eprops, term2id) = _construct_graph(term2id, facts)
end

function _construct_featured_graph(g, vprops, eprops, term2id)
	vertex_properties = sort(unique(reduce(vcat, (values(vprops)))))

	vpmap = Dict(reverse.(enumerate(vertex_properties)))
	nf = zeros(Float32, length(vertex_properties), nv(g))
	for (i, ps) in vprops
		foreach(p -> nf[vpmap[p], i] = 1, ps)
	end
	fg = FeaturedGraph(g)

	edge_properties = sort(unique(reduce(vcat, (values(eprops)))))
	epmap = Dict(reverse.(enumerate(edge_properties)))
	ef = zeros(Float32, length(edge_properties), ne(fg))
	for (i, (vi, vj)) in edges(fg)
		vj, vi, = Int64(vj), Int64(vi)
		e = (vi < vj) ? Edge(vi, vj) : Edge(Int64(vj), Int64(vi))
		ps = get(eprops, e, [])
		foreach(p -> ef[epmap[p], i] = 1, ps)
	end
	fg = FeaturedGraph(g; nf, ef)
end


end
