function BidirecationalL₂MiniBatch(pddld, domain::GenericDomain, problem::GenericProblem, plan::AbstractVector{<:Compound}; kwargs...)
	(L₂MiniBatch(pddld, domain, problem, plan; kwargs...),
	BackwardL₂MiniBatch(pddld, domain, problem, plan; kwargs...))
end

function BidirecationalLₛMiniBatch(pddld, domain::GenericDomain, problem::GenericProblem, plan::AbstractVector{<:Compound}; kwargs...)
	(LₛMiniBatch(pddld, domain, problem, plan; kwargs...),
	BackwardLₛMiniBatch(pddld, domain, problem, plan; kwargs...))
end 

function BidirecationalLgbfsMiniBatch(pddld, domain::GenericDomain, problem::GenericProblem, plan::AbstractVector{<:Compound}; kwargs...)
	(LgbfsMiniBatch(pddld, domain, problem, plan; kwargs...),
	BackwardLgbfsMiniBatch(pddld, domain, problem, plan; kwargs...))
end

function BidirecationalLRTMiniBatch(pddld, domain::GenericDomain, problem::GenericProblem, plan::AbstractVector{<:Compound}; kwargs...)
	(LRTMiniBatch(pddld, domain, problem, plan; kwargs...)	,
	BackwardLRTMiniBatch(pddld, domain, problem, plan; kwargs...)	)
end 
