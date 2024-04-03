
struct LazyVCatMatrix{T,N,M<:AbstractMatrix} <:AbstractMatrix{T}
	xs::NTuple{N,M}
	function LazyVCatMatrix(xs::NTuple{N,M}) where {N,M}
		n = size(first(xs),2)
		all(n == size(x,2) for x in xs) || error("all matrices should be of the same time")
		T = eltype(first(xs))
		all(T == eltype(x) for x in xs) || error("all matrices should have the same element type")
		new{T,N,M}(xs)
	end
end

function Base.size(A::LazyVCatMatrix)
	rows = mapreduce(x -> size(x,1), +, A.xs)
	cols = size(first(A.xs),2)
	(rows,cols)
end
Base.size(A::LazyVCatMatrix,i::Int) = i == 1 ? mapreduce(x -> size(x,1), +, A.xs) : size(first(A.xs),2)

LazyVCatMatrix(xs...) = LazyVCatMatrix(xs)

function *(A::Matrix{T}, B::LazyVCatMatrix{T, N, Matrix{T}}) where {T,N}
	o = similar(A, size(A,1), size(first(B.xs),2))
	x = first(B.xs)
	v = view(A, :, 1:size(x,1))
	LinearAlgebra.mul!(o, v, x)
	offset = size(x,1)
	for x in Base.tail(B.xs)
		v = view(A, :, offset+1:offset+size(x,1))
		LinearAlgebra.mul!(o, v, x, true, true)
		offset += size(x,1)
	end
	o
end
