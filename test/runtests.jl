using NeuroPlanner
using NeuroPlanner.PDDL
using NeuroPlanner.Flux
using NeuroPlanner.Mill
using NeuroPlanner.SymbolicPlanners
using NeuroPlanner.Accessors
using NeuroPlanner.ChainRulesCore
using NeuroPlanner: add_goalstate
using Test
using Random
using PlanningDomains

DOMAINS = ["briefcaseworld", "driverlog", "depot", "sokoban","ipc23_ferry", 
			"ipc23_rovers", "ipc23_blocksworld", "ipc23_floortile", 
			"ipc23_satellite", "ipc23_spanner", "ipc23_childsnack", 
			"ipc23_miconic", "ipc23_sokoban", "ipc23_transport", 
			"blocks", "ferry", "gripper", "npuzzle", "spanner", 
			"elevators_00", "elevators_11"]



DOMAINS = ["briefcaseworld", "driverlog", "depot", "sokoban","ipc23_ferry", 
			"ipc23_rovers", "ipc23_blocksworld", "ipc23_floortile", 
			"ipc23_satellite", "ipc23_spanner", "ipc23_childsnack", 
			"ipc23_miconic", "ipc23_sokoban", "ipc23_transport", 
			"blocks", "ferry", "gripper", "npuzzle", "spanner", 
			"elevators_00", "elevators_11"]

domain_path(s) = joinpath(pkgdir(NeuroPlanner),"test","problems",s*".pddl")
problem_path(s) = joinpath(pkgdir(NeuroPlanner),"test","problems",s*"_01.pddl")
plan_path(s) = joinpath(pkgdir(NeuroPlanner),"test","problems",s*"_01.plan")
function load_problem_domain(domain_name) 
	(load_domain(domain_path(domain_name)), 
		load_problem(problem_path(domain_name)),
	)
end

function load_plan(domain_name)
	lines = readlines(plan_path(domain_name))
	lines = filter(s -> s[1] != ';', lines)
	map(lines) do s
		p = Symbol.(split(s[2:end-1]," "))
		Compound(p[1], Const.(p[2:end]))
	end
end


_isapprox(a::Nothing, b::Nothing; kwargs...) = true
_isapprox(a::ZeroTangent, b::Nothing; kwargs...) = true
_isapprox(a::Number, b::Number; kwargs...) = isapprox(a,b;kwargs...)
_isapprox(a::NamedTuple,b::NamedTuple; kwargs...) = all(_isapprox(a[k], b[k]; kwargs...) for k in keys(a))
_isapprox(a::Tangent,b::NamedTuple; kwargs...) = all(_isapprox(a[k], b[k]; kwargs...) for k in keys(b))
_isapprox(a::Tuple, b::Tuple; kwargs...) = all(_isapprox(a[k], b[k]; kwargs...) for k in keys(a))
_isapprox(a::AbstractArray, b::AbstractArray; kwargs...) = all(_isapprox.(a,b;kwargs...))

include("dedu_matrix.jl")
include("lazyvcat.jl")
include("knowledge_base.jl")
include("datanode.jl")
include("modelnode.jl")
include("groupfacts.jl")
include("edgebuilder.jl")
include("integration.jl")
