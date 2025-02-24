using Parameters
using SymbolicPlanners: PathNode
using DataStructures: Queue
using OneHotArrays: onehotbatch

abstract type AbstractMinibatch end

include("l2loss.jl")
include("lstar.jl")
include("lgbfs.jl")
include("lrt.jl")
include("bellman.jl")
include("lstar_lmcut.jl")

########
#	dispatch for loss function
########
loss(model, xy::L₂MiniBatch,surrogate=softplus) = l₂loss(model, xy, surrogate)
loss(model, xy::LₛMiniBatch,surrogate=softplus) = lₛloss(model, xy, surrogate)
loss(model, xy::LgbfsMiniBatch,surrogate=softplus) = lgbfsloss(model, xy, surrogate)
loss(model, xy::LRTMiniBatch,surrogate=softplus) = lrtloss(model, xy, surrogate)
loss(model, xy::BellmanMiniBatch,surrogate=softplus) = bellmanloss(model, xy, surrogate)
loss(model, xy::Tuple,surrogate=softplus) = sum(map(x -> lossfun(model, x), xy), surrogate)
loss(model, xy::BinClassBatch, surrogate=softplus) = binclassloss(model, xy)


function minibatchconstructor(name)
	name == "l2" && return(L₂MiniBatch)
	name == "l₂" && return(L₂MiniBatch)
	name == "lstar" && return(LₛMiniBatch)
	name == "lₛ" && return(LₛMiniBatch)
	name == "lgbfs" && return(LgbfsMiniBatch)
	name == "lrt" && return(LRTMiniBatch)
	name == "bellman" && return(BellmanMiniBatch)
	name == "levinloss" && return(LevinMiniBatch)
	name == "newlstar" && return(LₛMiniBatchPossibleInequalities)
	name == "binclass" && return(BinClassBatch)
	error("unknown loss $(name)")
end