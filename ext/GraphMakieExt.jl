module GraphMakieExt


using NeuroPlanner, GraphMakie
using NeuroPlanner.StatsBase
using GraphMakie: Point2f
using NeuroPlanner.Graphs

using NeuroPlanner.SymbolicPlanners: PathNode, GenericState, LinkedNodeRef


number_of_parents(ref::LinkedNodeRef) = ref.next === nothing ? 1 : 1 + number_of_parents(ref.next)
number_of_parents(st::Dict{UInt64, PathNode{GenericState}}) = [number_of_parents(v.parent) for v in values(st)]


in_trajectory(v::PathNode, trajectory::Vector{UInt64}) = v.id ∈ trajectory
in_trajectory(v::PathNode, trajectories::Vector{Vector{UInt64}}) = any(in_trajectory(v, t) for t in trajectories)
in_trajectory(e::Tuple{PathNode, PathNode}, trajectories::Vector{Vector{UInt64}}) = any(in_trajectory(e, t) for t in trajectories)
in_trajectory(e::Tuple{PathNode, PathNode}, trajectory::Vector{UInt64}) = in_trajectory(e[1], trajectory) && in_trajectory(e[2], trajectory)

function NeuroPlanner.plot_digraph(st::Dict{UInt64, PathNode{GenericState}}; trajectory = Vector{UInt64}(), normal_edge = :gray, trajectory_edge = :red, multiple_parents = :gray, reverse_edge = :gray)
	# create the layout
	g, id2gid, gid2id = digraph(st)
	depths = countmap([i.path_cost for i in values(st)])
	layout = map(values(st)) do i 
		c = i.path_cost
		y = depths[c]
		depths[c] -= 2
		Point2f(4*c, y)
	end

	node_color = map(values(st)) do v
		in_trajectory(v, trajectory) && return(trajectory_edge)
		number_of_parents(v.parent) == 1 ? normal_edge : multiple_parents
	end

	edge_color = map(edges(g)) do e
		i, j = gid2id[e.src], gid2id[e.dst]
		si, sj = st[i], st[j]
		in_trajectory((si, sj), trajectory) && return(trajectory_edge)
		si.path_cost < sj.path_cost ? normal_edge : reverse_edge # highlight reverse edges
	end
	graphplot(g; layout = layout, node_color, edge_color) 
end


export plot_digraph

end