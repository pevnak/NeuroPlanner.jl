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


function muladd!(c::Matrix, a::Matrix, ii, α, β)
	size(a,1) == size(c,1) || error("c and a should have the same number of rows")
	@inbounds for (i, j) in enumerate(ii)
		for k in 1:size(a,1)
			c[k,i] = β*c[k,i] + α*a[k, j]
		end
	end
end

function *(A::Matrix{T}, B::LazyVCatMatrix{T, N, DeduplicatedMatrix{T}}) where {T,N}
	x = first(B.xs)
	v = view(A, :, 1:size(x,1))
	z = v * x.x
	o = similar(A, size(A,1), size(x,2))
	muladd!(o, z, x.ii, true, false)
	offset = size(x,1)
	for x in Base.tail(B.xs)
		v = view(A, :, offset+1:offset+size(x,1))
		z = v * x.x
		muladd!(o, z, x.ii, true, true)
		offset += size(x,1)
	end
	o
end
