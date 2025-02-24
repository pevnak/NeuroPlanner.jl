
struct L₂MiniBatch{X,Y} <: AbstractMinibatch
	x::X 
	y::Y
end

function L₂MiniBatch(pddle, trajectory::AbstractVector{<:GenericState})
	L₂MiniBatch(batch(map(pddle, trajectory)),
     collect(length(trajectory):-1:1),
     )
end


function L₂MiniBatch(pddld, domain::GenericDomain, problem::GenericProblem, trajectory::AbstractVector{<:GenericState}; goal_aware = true, goal_state = goalstate(pddld.domain, problem), kwargs...)
	pddle = goal_aware ? add_goalstate(pddld, problem, goal_state) : specialize(pddld, problem)
	L₂MiniBatch(pddle, trajectory)
end

function L₂MiniBatch(pddld, domain::GenericDomain, problem::GenericProblem, st::Union{Nothing,RSearchTree}, trajectory::AbstractVector{<:GenericState}; kwargs...)
	L₂MiniBatch(pddld, domain, problem, trajectory; kwargs...)
end

function L₂MiniBatch(pddld, domain::GenericDomain, problem::GenericProblem, plan::AbstractVector{<:Julog.Term}; kwargs...)
	state = initstate(domain, problem)
	trajectory = SymbolicPlanners.simulate(StateRecorder(), domain, state, plan)
	L₂MiniBatch(pddld, domain, problem, trajectory; kwargs...)
end


struct L₂Loss end 

l₂loss(model, x, y) = Flux.Losses.mse(vec(model(x)), y)
l₂loss(model, xy::L₂MiniBatch) = l₂loss(model, xy.x, xy.y)
l₂loss(model, xy::L₂MiniBatch, surrogate) = l₂loss(model, xy.x, xy.y)
l₂loss(model, mb::NamedTuple{(:minibatch,:stats)}) = l₂loss(model, mb.minibatch)

(l::L₂Loss)(args...) = l₂loss(args...)
