struct DeduplicatedMatrix{T} <:AbstractMatrix{T}
	x::Matrix{T}
	ii::Vector{Int}
end

function DeduplicatedMatrix(x::Matrix)
	mask, ii = find_duplicates(x)
	DeduplicatedMatrix(x[:,mask], ii)
end

DeduplicatedMatrix(x::DeduplicatedMatrix) = DeduplicatedMatrix(x.x, x.ii)

Base.size(x::DeduplicatedMatrix) = (size(x.x, 1), length(x.ii))
Base.getindex(x::DeduplicatedMatrix, i::Int, j::Int) = x.x[i, x.ii[j]]
Base.Matrix(x::DeduplicatedMatrix) = x.x[:,x.ii]

(m::Flux.Chain)(x::DeduplicatedMatrix) = DeduplicatedMatrix(m(x.x), x.ii)
(m::Flux.Dense)(x::DeduplicatedMatrix) = DeduplicatedMatrix(m(x.x), x.ii)

function ChainRulesCore.rrule(::Type{DeduplicatedMatrix}, a, ii)
    function dedu_pullback(Δbar)
    	δx = zeros(eltype(a), size(a))
    	for (i,j) in enumerate(ii)
    		δx[:,j] += Δbar[:,i]
    	end
    	NoTangent(), δx, NoTangent()
    end
    return DeduplicatedMatrix(a, ii), dedu_pullback
end
