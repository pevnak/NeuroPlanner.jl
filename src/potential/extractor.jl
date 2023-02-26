"""
LinearExtractor{D,CDO,N,M}

Is a simple implementation of a potential heuristics. In its esence, it 
takes a compiled state and folds it a vector. The potential heuristic 
is fixed for a problem instance. It is not transferable between problem 
instances.

"""
struct LinearExtractor{DO,CDO,N,NP,M, G}
	domain::DO
	c_domain::CDO
	array_props::NTuple{N, Symbol}
	array_offsets::NTuple{NP, Int}
	scalar_props::NTuple{M, Symbol}
	scalar_offsets::NTuple{M, Int}
	dimension::Int
	goal::G
end

LinearExtractorGoalLess =  LinearExtractor{<:Any,<:Any,<:Any,<:Any,<:Any,<:Nothing}
LinearExtractorGoalAware = LinearExtractor{<:Any,<:Any,<:Any,<:Any,<:Any,<:AbstractVector}
LinearExtractorBlank = LinearExtractor{<:Any,<:Any,0,0,0,<:Nothing}

"""

"""
function LinearExtractor(domain, problem, c_domain, c_state; embed_goal = true)
	props = propertynames(c_state)
	array_props = filter(k -> getproperty(c_state, k) isa AbstractArray, props)
	scalar_props = filter(k -> getproperty(c_state, k) isa Number, props)
	l = map(k -> length(getproperty(c_state, k)), array_props)
	array_offsets = (0, cumsum(l)...)
	scalar_offsets = tuple([array_offsets[end] + i for i in eachindex(scalar_props)]...)
	dimension = isempty(scalar_offsets) ? array_offsets[end] : scalar_offsets[end]
	le = LinearExtractor(domain, c_domain, array_props, array_offsets, scalar_props, scalar_offsets, dimension, nothing)
	embed_goal ? add_goalstate(le, problem) : le
end

function initproblem(lx::LinearExtractor, problem; add_goal = true)
	lx, PDDL.compilestate(lx.c_domain, initstate(lx.domain, problem))
end

function add_goalstate(lx::LinearExtractor, problem::GenericProblem)
	add_goalstate(lx, problem, goalstate(lx.domain, problem))
end

function add_goalstate(lx::LinearExtractor, problem, goal)
	LinearExtractor(
		lx.domain,
		lx.c_domain,
		lx.array_props,
		lx.array_offsets,
		lx.scalar_props,
		lx.scalar_offsets,
		lx.dimension,
		state2vec(Float32, lx, goal),
	)
end

"""
state2vec(T::DataType, e::LinearExtractor, s::CompiledState)

Convert state to vector. Goal is not added.
"""
function state2vec(T::DataType, e::LinearExtractor, s::CompiledState)
	x = zeros(T, e.dimension)
	for (i,k) in enumerate(e.array_props)
		start = e.array_offsets[i] + 1
		stop = e.array_offsets[i + 1]
		x[start:stop] .= vec(getproperty(s, k))
	end
	for (i, k) in enumerate(e.scalar_props)
		x[e.scalar_offsets[i]] = getproperty(s, k)
	end
	x
end

function state2vec(T::DataType, e::LinearExtractor, s::State)
	state2vec(T, e, PDDL.compilestate(e.c_domain, s))
end

function (e::LinearExtractorGoalLess)(T::DataType, s::CompiledState)
	state2vec(T, e, s)
end

function (e::LinearExtractorGoalAware)(T::DataType, s::CompiledState)
	vcat(state2vec(T, e, s), e.goal)
end

(e::LinearExtractor)(T::DataType, s::State) = e(T, PDDL.compilestate(e.c_domain, s))
(e::LinearExtractor)(s::State) = e(Float32, s)
