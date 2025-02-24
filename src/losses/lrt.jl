
#########
#	LRT loss enforcing ordering on the trajectory
#   Caelan Reed Garrett, Leslie Pack Kaelbling, and Tomás Lozano-Pérez. Learning to rank for synthesizing planning heuristics. page 3089–3095, 2016.
#########
struct LRTMiniBatch{X,H,Y} <: AbstractMinibatch
	x::X 
	H₊::H 
	H₋::H 
	path_cost::Y
	sol_length::Int64
end

function LRTMiniBatch(pddle, trajectory::AbstractVector{<:GenericState})
	n = length(trajectory)
	if n < 2
		H₊ = onehotbatch([], 1:n)
		H₋ = onehotbatch([], 1:n)
	else
		H₊ = onehotbatch(1:n -1 , 1:n)
		H₋ = onehotbatch(2:n, 1:n)
	end
	LRTMiniBatch(batch(map(pddle, trajectory)),
		H₊,
		H₋,
		collect(length(trajectory):-1:1),
		n
     )
end

function LRTMiniBatch(pddld, domain::GenericDomain, problem::GenericProblem, st::Union{Nothing,RSearchTree}, trajectory::AbstractVector{<:GenericState}; kwargs...)
	LRTMiniBatch(pddld, domain, problem, trajectory; kwargs...)
end

function LRTMiniBatch(pddld, domain::GenericDomain, problem::GenericProblem, plan::AbstractVector{<:Julog.Term}; kwargs...)
	state = initstate(domain, problem)
	trajectory = SymbolicPlanners.simulate(StateRecorder(), domain, state, plan)
	LRTMiniBatch(pddld, domain, problem, trajectory; kwargs...)
end

function LRTMiniBatch(pddld, domain::GenericDomain, problem::GenericProblem, trajectory; goal_aware = true, max_branch = typemax(Int), goal_state = goalstate(pddld.domain, problem), kwargs...)
	pddle = goal_aware ? add_goalstate(pddld, problem, goal_state) : specialize(pddld, problem)
	LRTMiniBatch(pddle, trajectory)
end

function lrtloss(model, x, g, H₊, H₋, surrogate = softplus)
	f = model(x)
	o = f * H₋ - f * H₊
	isempty(o) && return(zero(eltype(o)))
	mean(surrogate.(o))
end

lrtloss(model, xy::LRTMiniBatch, surrogate = softplus) = lrtloss(model, xy.x, xy.path_cost, xy.H₊, xy.H₋)
