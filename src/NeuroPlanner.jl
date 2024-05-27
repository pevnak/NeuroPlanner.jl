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
using SparseArrays
using LinearAlgebra
using StaticBitSets


"""
initproblem(ex, problem; add_goal = true)

Specialize extractor for the given problem instance and return init state 
"""
function initproblem(ex, problem; add_goal=true)
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
export KBEntry, KnowledgeBase, append, atoms

#include("hyper/mha.jl")
include("relational/deduplication.jl")
include("relational/coloring.jl")
export deduplicate, color
include("relational/dedu_matrix.jl")
include("relational/mha.jl")
export MultiheadAttention
include("relational/renamer.jl")
include("relational/edgebuilder.jl")
export EdgeBuilder

# a basic architecture based on hyper-graph representatation of predicates
include("lrnn/pure_extractor.jl")
export LRNN

# Object Binary structures by Sir and Horcik
include("object_binary/groupfacts.jl")
include("object_binary/object_binary.jl")
export ObjectBinary, ObjectBinaryFE, ObjectBinaryFENA, ObjectBinaryME
include("object_binary/atom_binary.jl")
export AtomBinary, AtomBinaryFE, AtomBinaryFENA, AtomBinaryME
include("object_binary/object_pair.jl")
export ObjectPair
include("object_binary/object_atom.jl")
export ObjectAtom
include("object_binary/object_atom_bip.jl")
export ObjectAtomBip, ObjectAtomBipFE, ObjectAtomBipFENA, ObjectAtomBipME

# ASNet and HGNN by Silvia
include("asnets/extractor.jl")
export ASNet

#include("graphkernel/graphkernel.jl")
#export GraphKernel

# ASNet is a pain in ...
include("levin_asnet/extractor.jl")
include("levin_asnet/loss.jl")
include("levin_asnet/bfs_planner.jl")
export LevinASNet, BFSPlanner

include("hgnn/extractor.jl")
export HGNNLite, HGNN

include("admissible_planner/extractor.jl")
export AdmissibleExtractor

include("admissible_planner/apply_policy.jl")
export ApplyPolicy, setOutputToPolicy, roundPolicyOutput

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

include("utils/utils.jl")
export ffnn, tblogger, W15AStarPlanner, W20AStarPlanner, dedup_fmb

include("utils/training.jl")
export train!, sample_minibatch, prepare_minibatch, isvalid, isinvalid

include("utils/solution_tracking.jl")
export solve_problem, update_solutions!, update_solution, issolved, show_stats, _show_stats
include("utils/problems.jl")
export load_plan, save_plan, plan_file, setup_problem, setup_classic, getproblem, accomodate_leah_plans, merge_ferber_problems, hashfile, _parse_plan, systematize, similarity_of_problems
end
