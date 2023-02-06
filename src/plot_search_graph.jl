SearchTree{N} = Dict{UInt64, N} where {N<:SymbolicPlanners.PathNode}

"""
dt = function depth(st::SearchTree)

compute mapping of node_id to depth in search tree

```julia	
julia> dt = depth(st)
Dict{UInt64, Int64} with 7 entries:
  0x9d346f48836060c5 => 0
  0x239772ea48f56dd5 => 1
  0xd6f72130b34ab51a => 1
  0x34210d401741745b => 2
  0x097fd218ea0f517a => 2
  0xa8eb4cb52221b2ae => 2
  0xd5dff6d6dad04750 => 3
```

"""
function depth(st::SearchTree)
	dt = Dict{UInt64, Int64}()
	for s in keys(st)
		updatedepth!(dt::Dict{UInt64, Int64}, st::SearchTree, s)
	end
	dt
end

function updatedepth!(dt::Dict{UInt64, Int64}, st::SearchTree, s)
	haskey(dt, s) && return(dt[s])
	node = st[s]
	if node.parent_id === nothing 
		dt[s] = 0 
	else
		dt[s] = updatedepth!(dt, st, node.parent_id) + 1
	end
end

"""
searchtree2graph(st)

convert searchtree to graph

```julia
julia> searchtree2graph(st)
{2752, 2751} directed simple Int64 graph
```
"""
function searchtree2graph(st)
	d = Dict(reverse.(enumerate(keys(st))))
	g = DiGraph(length(d))
	for s in values(st)
		s.parent_id === nothing && continue
		s.id === nothing && continue
		add_edge!(g, d[s.parent_id], d[s.id])
	end
	g
end

function treelayers(st)
	dt = depth(st)
	max_depth = maximum(values(dt))
	max_width = maximum(values(countmap(values(dt))))
	nodeids = collect(keys(dt))

	layers = Vector{Vector{UInt64}}()
	# plot root node
	root_node = only(filter(k -> dt[k] == 0, nodeids))
	push!(layers, [root_node])

	parent_positions = Dict(root_node => 0)
	for d in 1:max_depth
		childs = collect(filter(k -> dt[k] == d, nodeids))
		pp = [parent_positions[st[k].parent_id] for k in childs]
		childs = childs[sortperm(pp)]
		parent_positions = Dict(reverse.(enumerate(childs)))
		push!(layers, childs)
	end
	dt, layers
end


function plot_search_tree()
	dt, layers = treelayers(st)
end

preamble = """
\\documentclass{standalone}
\\usepackage{tikz}
\\usepackage{verbatim}
\\usepackage{adjustbox}
\\usetikzlibrary{arrows,shapes}


\\begin{document}

\\tikzstyle{nonoptimalnode}=[circle,fill=black!25,minimum size=20pt,inner sep=0pt]
\\tikzstyle{optimalnode} = [nonoptimalnode, fill=red!24]
\\tikzstyle{nonoptimaledge} = [draw,->,thin]
\\tikzstyle{optimaledge} = [draw,thick,->]
\\begin{tikzpicture}
"""

closing = """
\\end{tikzpicture}
\\end{document}
"""

function plot_search_tree(io::IOStream, st::SearchTree, trajectory::AbstractSet, hvals = Dict{UInt64, Float64}(), onlystates = nothing;α = 1, β = 1)
	exportstate(s, onlystates::Nothing) = true
	exportstate(s, onlystates) = s ∈ onlystates
	dt, layers = treelayers(st)
	# k = max(length(layers), maximum(length.(layers)))
	println(io, preamble)
	# println(io, "\\begin{tikzpicture}[scale=$(1/k)")
	for (row, states) in enumerate(layers)
		offset = div(length(states),2)
		for (col, state) in enumerate(states)
			!exportstate(state, onlystates) && continue
			pos = "($(col - offset) ,- $(row))"
			label = "node$(state)"
			style = state ∈ trajectory ? "optimal" : "nonoptimal"
			f = β * get(hvals, state, 0) + α* st[state].path_cost
			f = round(f, digits = 2)
			println(io, "\\node[$(style)node] ($label) at $(pos) {$(f)};")
			pid = st[state].parent_id
			if pid !== nothing 
				println(io, "\\path[$(style)edge] (node$(pid).south) -- ($label.north);")
			end
		end
	end
	println(io, closing)
end

function plot_search_tree(filename::String, st::SearchTree, trajectory, onlystates = nothing)
	open(io -> plot_search_tree(io, st, trajectory, onlystates), filename, "w")
end


"""
struct EvalTracker{H<:Heuristic} <: Heuristic 
	heuristic::H
	order::Dict{UInt64,Int}
end

tructure to log the order at which the state was evaluated first time

"""
struct EvalTracker{H<:Heuristic} <: Heuristic 
	heuristic::H
	order::Dict{UInt64,Int}
	vals::Dict{UInt64,Float64}
end

EvalTracker(heuristic) = EvalTracker(heuristic, Dict{UInt64,Int}(), Dict{UInt64,Float64}())

#reexport the heuristic api
Base.hash(g::EvalTracker, h::UInt) = hash(g.model, hash(g.pddle, h))

function SymbolicPlanners.compute(h::EvalTracker, domain::Domain, state::State, spec::Specification)
	id = hash(state)
	get!(h.order, id, length(h.order) + 1)
	get!(h.vals, id, SymbolicPlanners.compute(h.heuristic, domain, state, spec))
end

SymbolicPlanners.precompute!(h::EvalTracker, domain::Domain, state::State, spec::Specification) = SymbolicPlanners.precompute!(h.heuristic, domain, state, spec)
# SymbolicPlanners.ensure_precompute!(h::EvalTracker, args) = SymbolicPlanners.ensure_precompute!(h.heuristic, args)
# SymbolicPlanners.is_precomputed!(h::EvalTracker) = SymbolicPlanners.is_precomputed!(h.heuristic)



function plot_example()
	problem_name = "ferry"
	domain_pddl, problem_files = getproblem(problem_name, true)[1:2]
	domain = load_domain(domain_pddl)
	problem = load_problem(problem_files[end])


	state = initstate(domain, problem)
	spec = MinStepsGoal(problem)
	# h = EvalTracker(HMax())
	h = EvalTracker(HAdd())
	planner = AStarPlanner(h; max_time=3600, save_search = true)

	goal = PDDL.get_goal(problem)
	t = @elapsed sol = planner(domain, state, spec)


	# to plot the full tree
	# prefix = "/Users/tomas.pevny/Work/Presentations/planning/animation"
	prefix = "/tmp"
	plot_search_tree(prefix*"/debug.tex", sol.search_tree,  Set(hash.(sol.trajectory)), h.vals)

	# plot the searc incrementally
	search_order = map(first, sort(collect(h.order), lt = (i,j) -> i[2] < j[2]))
	# show the first 100 steps of the search
	for i in 1:200
		plot_search_tree("$(prefix)/debug_$(i).tex", sol.search_tree,  Set(hash.(sol.trajectory)), search_order[1:i])
	end
end