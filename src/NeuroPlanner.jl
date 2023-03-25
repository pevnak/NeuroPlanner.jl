module NeuroPlanner

using PDDL
using Julog
using Graphs
using Flux
using OneHotArrays
using Statistics
using SymbolicPlanners
using StatsBase
using GraphNeuralNetworks
using Mill
using Setfield
using MLUtils
using NeuralAttentionlib
using HierarchicalUtils
using ChainRulesCore

"""
initproblem(pddld::HyperExtractor{<:Nothing,<:Nothing}, problem; add_goal = true)

Specialize extractor for the given problem instance and return init state 
"""
function initproblem(pddld, problem; add_goal = true)
	pddle = add_goal ? add_goalstate(pddld, problem) : specialize(pddld, problem)
	pddle, initstate(pddld.domain, problem)
end


include("relational/knowledge_base.jl")
include("relational/knowledge_model.jl")
export KBEntry, KnowledgeBase, append
include("hyper/extractor.jl")
include("hyper/deduplication.jl")
export HyperExtractor, deduplicate

include("hyper/mha.jl")
export MultiheadAttention

include("multigraph/extractor.jl")
export PDDLExtractor, MultiGraph, MultiGNN, FeaturedMultiGraph, MultiModel, initproblem

include("potential/extractor.jl")
export LinearExtractor

include("rsearch_tree.jl")
include("losses.jl")
export L₂MiniBatch, LₛMiniBatch, LRTMiniBatch, LgbfsMiniBatch

include("artificial_goals.jl")
include("sample_trace.jl")
export sample_trace, sample_forward_trace, sample_backward_trace, sample_backward_tree, search_tree_from_trajectory
export BackwardSampler
include("heuristic.jl")
export NeuroHeuristic

MLUtils.batch(xs::AbstractVector{<:AbstractMillNode}) = reduce(catobs, xs)
end
