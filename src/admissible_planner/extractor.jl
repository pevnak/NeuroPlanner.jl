struct AdmissibleEXtractor8{G,EX,PR}
	domain::G
	inner_ex::EX
	problem::PR
end

function AdmissibleEXtractor8(domain::GenericDomain; hnet=HyperExtractor, kwargs...)
	#model_params = (;message_passes, residual, lite = false)
	#AdmissibleEXtractor8(domain, nothing, model_params, nothing, nothing, nothing, nothing,nothing)
	AdmissibleEXtractor8(domain, hnet(domain; kwargs...), nothing)
end

function AdmissibleEXtractor8(extracator; problem=nothing)
	AdmissibleEXtractor8(extracator.domain, extracator, problem)
end

AdmissibleExtractor = AdmissibleEXtractor8

isspecialized(ex::AdmissibleEXtractor8) = isspecialized(ex.inner_ex) && ex.problem !== nothing
hasgoal(ex::AdmissibleEXtractor8) = hasgoal(ex.inner_ex)

function Base.show(io::IO, ex::AdmissibleExtractor)
	s = isspecialized(ex) ? "Specialized" : "Unspecialized"
	s *=" AdmissibleEXtractor8 using $(typeof(ex.inner_ex)) for $(ex.inner_ex.domain.name )"
	print(io, s)
end

function specialize(ex::AdmissibleEXtractor8, problem)
	inner_ex_spec = specialize(ex.inner_ex, problem)
	pddle = AdmissibleEXtractor8(inner_ex_spec; problem)
	return pddle
end

function (ex::AdmissibleExtractor)(state)
	isspecialized(ex) || "Err: Performing extraction with Unspecialized AdmissibleExtractor"
	kb = ex.inner_ex(state)
	
	#hmax = HMax()
	#heurTuple = [hmax(ex.inner_ex.domain, state, ex.problem) ; 0;;]
	#optHeur =  Float32(ex.ch(state))
	lm = LM_CutHeuristic()
	lm_val = lm(ex.domain, state, Specification(ex.problem))
	#@show lm_val
	heuristics = [lm_val ; 0f0;;]
	#@show typeof(kb) <: KnowledgeBase{KS,VS} where {KS,VS}
	kb = prepend(kb, :h, heuristics)
	#@show typeof(kb) <: KnowledgeBase{KS,VS} where {KS,VS}
	return kb
end

function add_goalstate(ex::AdmissibleEXtractor8, problem, goal = goalstate(ex.inner_ex.domain, problem))
	AdmissibleEXtractor8(add_goalstate(ex.inner_ex, problem, goal); problem)
end

function add_initstate(ex::AdmissibleEXtractor8, problem, init = initstate(ex.inner_ex.domain, problem))
	AdmissibleEXtractor8(add_initstate(ex.inner_ex, problem, init); problem)
end

