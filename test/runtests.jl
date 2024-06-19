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
