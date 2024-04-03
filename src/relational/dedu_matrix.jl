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

function muladd!(c::Matrix, a::Matrix, ii, α, β)
	size(a,1) == size(c,1) || error("c and a should have the same number of rows")
	@inbounds for (i, j) in enumerate(ii)
		for k in 1:size(a,1)
			c[k,i] = β*c[k,i] + α*a[k, j]
		end
	end
end

function ChainRulesCore.rrule(::Type{DeduplicatedMatrix}, a, ii)
    function dedu_pullback(ȳ)
    	δx = similar(a)
    	muladd!(δc, ȳ, ii, true,false)
    	NoTangent(), δx, NoTangent()
    end
    return DeduplicatedMatrix(a, ii), dedu_pullback
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

function ChainRulesCore.rrule(::typeof(*), A::Matrix{T}, B::LazyVCatMatrix{T, N, DeduplicatedMatrix{T}}) where {T,N}
    function lazymull_pullback(ȳ)
    	Ȳ = unthunk(ȳ)
    	dA = @thunk(Ȳ * B')
    	dB = @thunk(project_B(A' * Ȳ))
    	for (i,j) in enumerate(ii)
    		δx[:,j] += ȳ[:,i]
    	end
    	NoTangent(), δx, NoTangent()
    end
    return A*B, lazymull_pullback
end

# function rrule(
#     ::typeof(*),
#     A::AbstractVecOrMat{<:CommutativeMulNumber},
#     B::AbstractVecOrMat{<:CommutativeMulNumber},
# )
#     project_A = ProjectTo(A)
#     project_B = ProjectTo(B)
#     function times_pullback(ȳ)
#         Ȳ = unthunk(ȳ)
#         dA = @thunk(project_A(Ȳ * B'))
#         dB = @thunk(project_B(A' * Ȳ))
#         return NoTangent(), dA, dB
#     end
#     return A * B, times_pullback
# end


