using PDDL2Graph
using PDDL
using Graphs
using GraphPlot
using Cairo
using Compose


domain = load_domain("sokoban.pddl")
problem = load_problem("s1.pddl")

domain = load_domain("benchmarks/blocks-slaney/domain.pddl")
problem = load_problem("benchmarks/blocks-slaney/blocks10/task01.pddl")

domain = load_domain("benchmarks/ferry/ferry.pddl")
problem = load_problem("benchmarks/ferry/train/ferry-l2-c1.pddl")

domain = load_domain("benchmarks/gripper/domain.pddl")
problem = load_problem("benchmarks/gripper/problems/gripper-n1.pddl")

domain = load_domain("benchmarks/n-puzzle/domain.pddl")
problem = load_problem("benchmarks/n-puzzle/train/n-puzzle-2x2-s1.pddl")

# domain = load_domain("benchmarks/zenotravel/domain.pddl")
# problem = load_problem("benchmarks/zenotravel/train/zenotravel-cities2-planes1-people2-1864.pddl")


state = initstate(domain, problem)
goalstate(domain, problem)

a = PDDL2Graph.state2graph(state)
b = PDDL2Graph.goal2graph(a.term2id, problem)

function graph_labels(a)
	map(sort(reverse.(collect(a.term2id)), lt = (i,j) -> i[1] < j[1])) do i 
		s = string(i[2])
		s *= haskey(a.vprops, i[1]) ? " ("*join(string.(a.vprops[i[1]]))*")" : ""
		s
	end
end
locs_x, locs_y = circular_layout(a.graph)
# locs_x, locs_y = spring_layout(a.graph)
nodelabel = graph_labels(a)
edgelabel = [join(a.eprops[e]) for e in edges(a.graph)]
gp = gplot(a.graph, locs_x, locs_y; nodelabel, edgelabel);
draw(SVG("/tmp/state.svg", 16cm, 16cm), gp)
nodelabel = graph_labels(b)
edgelabel = [join(b.eprops[e]) for e in edges(b.graph)]
gp = gplot(b.graph, locs_x, locs_y; nodelabel, edgelabel);
draw(SVG("/tmp/goal.svg", 16cm, 16cm), gp)
