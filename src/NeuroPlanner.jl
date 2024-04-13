module NeuroPlanner

using PDDL
using Julog
using Graphs
using Flux
using OneHotArrays
using Statistics
using SymbolicPlanners
using StatsBase
using Combinatorics
using Mill
using MLUtils
using DataStructures
using HierarchicalUtils
using ChainRulesCore
using Accessors
using LinearAlgebra


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

#####
#	A general support for architectures
#####
include("utils.jl")
include("mill_extension/mill_extension.jl")
include("relational/knowledge_base.jl")
include("relational/knowledge_model.jl")
export KBEntry, KnowledgeBase, append
include("relational/deduplication.jl")
export deduplicate
include("relational/dedu_matrix.jl")
include("relational/mha.jl")
export MultiheadAttention
include("relational/renamer.jl")
include("relational/edgebuilder.jl")

# a basic architecture based on hyper-graph representatation of predicates
include("lrnn/pure_extractor.jl")
include("lrnn/mixed_extractor.jl")
include("lrnn/mixed_extractor2.jl")
include("lrnn/mixed_extractor3.jl")
export MixedLRNN, MixedLRNN2, MixedLRNN3, LRNN 

# Object Binary structures by Sira and Horcik
include("object_binary/object_binary.jl")
include("object_binary/atom_binary.jl")
include("object_binary/object_pair.jl")
export ObjectBinary, AtomBinary, ObjectPair

# ASNet and HGNN by Silvia
include("asnets/extractor.jl")
export ASNet
include("hgnn/extractor.jl")
export HGNNLite, HGNN

# ASNet is a pain in ...
include("levin_asnet/extractor.jl")
include("levin_asnet/loss.jl")
include("levin_asnet/bfs_planner.jl")
export LevinASNet, BFSPlanner

# Potential heuristic is useless
include("potential/extractor.jl")
export LinearExtractor

# loss functions
include("rsearch_tree.jl")
include("losses.jl")
export L₂MiniBatch, LₛMiniBatch, LRTMiniBatch, LgbfsMiniBatch

# leftovers from a small research in artificial goals
include("artificial_goals.jl")
include("sample_trace.jl")
export sample_trace, sample_forward_trace, sample_backward_trace, sample_backward_tree, search_tree_from_trajectory
export BackwardSampler
include("heuristic.jl")
export NeuroHeuristic

export add_goalstate, add_initstate, specialize

MLUtils.batch(xs::AbstractVector{<:AbstractMillNode}) = reduce(catobs, xs)
end
