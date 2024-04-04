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

function scatter_cols!(c::Matrix, a::AbstractMatrix, ii, α, β)
	size(a,1) == size(c,1) || error("c and a should have the same number of rows")
	size(a,2) ≤ maximum(ii) || error("a has to has to have at least $(maximum(ii)) columns")
	size(c,2) ≤ length(ii) || error("c has to has to have at least $(length(ii)) columns")
	@inbounds for (i, j) in enumerate(ii)
		for k in 1:size(a,1)
			c[k,i] = β*c[k,i] + α*a[k, j]
		end
	end
end

function gather_cols!(c::Matrix, a::AbstractMatrix, ii, α, β)
	size(a,1) == size(c,1) || error("c and a should have the same number of rows")
	size(c,2) ≤ maximum(ii) || error("c has to has to have at least $(maximum(ii)) columns")
	size(a,2) ≤ length(ii) || error("a has to has to have at least $(length(ii)) columns")
	@inbounds for (i, j) in enumerate(ii)
		for k in 1:size(a,1)
			c[k,j] = β*c[k,j] + α*a[k, i]
		end
	end
end

function ChainRulesCore.rrule(::Type{DeduplicatedMatrix}, a, ii)
    function dedu_pullback(ȳ)
    	Ȳ = unthunk(ȳ)
    	δx = zeros(eltype(a), size(a))
    	gather_cols!(δx, ȳ, ii, true,true)
    	NoTangent(), δx, NoTangent()
    end

    function dedu_pullback(ȳ::Tangent{Any, @NamedTuple{x::Matrix{Float32}, ii::ZeroTangent}})
    	Ȳ = unthunk(ȳ)
    	# @show typeof(Ȳ)
    	δx = Ȳ.x
    	NoTangent(), δx, NoTangent()
    end
    return DeduplicatedMatrix(a, ii), dedu_pullback
end


function *(A::Matrix{T}, B::LazyVCatMatrix{T, N, DeduplicatedMatrix{T}}) where {T,N}
	x = first(B.xs)
	v = view(A, :, 1:size(x,1))
	z = v * x.x
	o = similar(A, size(A,1), size(x,2))
	scatter_cols!(o, z, x.ii, true, false)
	offset = size(x,1)
	for x in Base.tail(B.xs)
		v = view(A, :, offset+1:offset+size(x,1))
		z = v * x.x
		scatter_cols!(o, z, x.ii, true, true)
		offset += size(x,1)
	end
	o
end


function *(A::Matrix{T}, B::LinearAlgebra.Adjoint{T,LazyVCatMatrix{T, N, DeduplicatedMatrix{T}}}) where {T,N}
	o = similar(A, size(A,1), size(B,2))
	offset = 0
	for x in B.parent.xs
		v = view(o, :, offset+1:offset+size(x,1))
		subA = zeros(T, size(A,1), size(x.x,2))
		gather_cols!(subA, A, x.ii, true, true)
		LinearAlgebra.mul!(v, subA, x.x')
		offset += size(x,1)
	end
	o
end

function ChainRulesCore.ProjectTo(B::LazyVCatMatrix{T,N,DeduplicatedMatrix{T}}) where {T,N}
	sizes = map(x -> size(x.x), B.xs)
	indices = map(x -> x.ii, B.xs)
	element = eltype(B)
	return(ProjectTo{LazyVCatMatrix{T,N,DeduplicatedMatrix{T}}}(;sizes, indices, element))
end

function (a::ChainRulesCore.ProjectTo{LazyVCatMatrix{T,N,DeduplicatedMatrix{T}}})(x::Matrix) where {T,N}
	offset = 0 
	xs = map(1:length(a.sizes)) do i
		rows, cols = a.sizes[i]
		start = offset + 1
		stop = offset + rows
		offset += rows
		sub_δ = x[start:stop, :]
		δ = zeros(T, rows, cols)
		gather_cols!(δ, sub_δ, a.indices[i], true, true)
		Tangent{DeduplicatedMatrix{T}}(;x = δ)
	end
	return(Tangent{LazyVCatMatrix{T,N,DeduplicatedMatrix{T}}}(;xs))
end
