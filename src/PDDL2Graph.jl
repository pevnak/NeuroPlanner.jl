module PDDL2Graph

using PDDL
using Julog
using Graphs


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

end
