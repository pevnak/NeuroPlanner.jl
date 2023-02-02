"""
LinearExtractor{D,CDO,N,M}

Is a simple implementation of a potential heuristics. In its esence, it 
takes a compiled state and folds it a vector. The potential heuristic 
is fixed for a problem instance. It is not transferable between problem 
instances.

"""
struct LinearExtractor{D,CDO,N,M}
	domain::DO
	c_domain::CDO
	array_props::NTuple{N, Symbol}
	array_offsets::NTuple{N + 1, Int}
	scalar_props::NTuple{M, Symbol}
	scalar_offsets::NTuple{M, Int}
	dimension::Int
end



function LinearExtractor(domain, problem; kwargs...)
	c_domain, c_state = compiled(domain, initstate(domain, problem))
	props = propertynames(c_state)
	array_props = filter(k -> getproperty(c_state, k) isa AbstractArray, props)
	scalar_props = filter(k -> getproperty(c_state, k) isa Number, props)
	l = map(k -> length(getproperty(c_state, k)), array_props)
	array_offsets = (0, cumsum(l)...)
	scalar_offsets = tuple([array_offsets[end] + i for i in eachindex(scalar_props)]...)
	dimension = isempty(scalar_offsets) ? array_offsets[end] : scalar_offsets[end]
	LinearExtractor(domain, c_domain, array_props, array_offsets, scalar_props, scalar_offsets, dimension)
end


function (e::LinearExtractor)(T, s::CompiledState)
	x = zeros(T, dimension)
	for (i,k) in enumerate(array_props)
		start = array_offsets[i] + 1
		stop = array_offsets[i + 1]
		x[start:stop] .= vec(getproperty(s, k))
	end
	for (i, k) in enumerate(scalar_props)
		x[scalar_offsets[i]] = getproperty(s, k)
	end
end

function (e::LinearExtractor)(T, s::State) = e(T, compile(e.c_domain, s))
