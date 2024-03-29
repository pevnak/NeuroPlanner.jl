module NeuroPlanner

using PDDL
using Julog
using Graphs
using Flux
using OneHotArrays
using Statistics
using SymbolicPlanners
using StatsBase
using Mill
using MLUtils
using DataStructures
using HierarchicalUtils
using ChainRulesCore
using Accessors

"""
initproblem(ex, problem; add_goal = true)

Specialize extractor for the given problem instance and return init state 
"""
function initproblem(ex, problem; add_goal = true)
	ex = specialize(ex, problem)
	pddle = add_goal ? add_goalstate(ex, problem) : ex
	pddle, initstate(ex.domain, problem)
end
export initproblem


include("relational/knowledge_base.jl")
include("relational/knowledge_model.jl")
export KBEntry, KnowledgeBase, append
include("hyper/extractor.jl")
include("hyper/deduplication.jl")
include("hyper/dedu_matrix.jl")
export HyperExtractor, deduplicate
include("hyper/mha.jl")
export MultiheadAttention

include("asnets/extractor.jl")
export ASNet

include("levin_asnet/extractor.jl")
include("levin_asnet/loss.jl")
include("levin_asnet/bfs_planner.jl")
export LevinASNet, BFSPlanner

include("hgnn/extractor.jl")
export HGNNLite, HGNN

include("potential/extractor.jl")
export LinearExtractor

include("rsearch_tree.jl")

include("forward_losses.jl")
export L₂MiniBatch, LₛMiniBatch, LRTMiniBatch, LgbfsMiniBatch
include("backward_losses.jl")
export BackwardL₂MiniBatch, BackwardLₛMiniBatch, BackwardLRTMiniBatch, BackwardLgbfsMiniBatch
include("bidirectional_losses.jl")

include("artificial_goals.jl")
include("sample_trace.jl")
export sample_trace, sample_forward_trace, sample_backward_trace, sample_backward_tree, search_tree_from_trajectory
export BackwardSampler
include("heuristic.jl")
export NeuroHeuristic

export add_goalstate, add_initstate, specialize

MLUtils.batch(xs::AbstractVector{<:AbstractMillNode}) = reduce(catobs, xs)
end
