"""
LazyVCatMatrix

A simple for lazy concantenation of matrices. The intended usage
	is in ProductNode, where individual models outputs matrices,
	which are the concatenated and passed to Dense Layer. That said
	`LazyVCatMatrix` implements only matrix multiplication and does not
	have at the moment `getindex` and `setindex!`.

The indended usage is something like
```julia
x = randn(16,127)
y = randn(4,127)
z = randn(8,127)
w = randn(16,28)
L = NeuroPlanner.LazyVCatMatrix((x,y,z));
A = vcat(x,y,z)

w * A ≈ w * L
```
"""
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

function Base.show(io::IO, ::MIME{Symbol("text/plain")}, a::LazyVCatMatrix{T,N,M}) where {T,N,M}
	rows = join(map(x -> size(x,1) , a.xs),",")
	println(io,"LazyVCatMatrix{$(T)} ($(rows)=$(size(a,1)))×$(size(a,2))")
end

function Base.size(A::LazyVCatMatrix)
	rows = mapreduce(x -> size(x,1), +, A.xs)
	cols = size(first(A.xs),2)
	(rows,cols)
end
Base.size(A::LazyVCatMatrix,i::Int) = i == 1 ? mapreduce(x -> size(x,1), +, A.xs) : size(first(A.xs),2)
Base.eltype(A::LazyVCatMatrix{T,N,M}) where {T,N,M} = T

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

function *(A::Matrix{T}, B::LinearAlgebra.Adjoint{T, LazyVCatMatrix{T, N, Matrix{T}}}) where {T,N}
	o = similar(A, size(A,1), size(B,2))
	offset = 0
	for x in B.parent.xs
		v = view(o, :, offset+1:offset+size(x,1))
		LinearAlgebra.mul!(v, A, x')
		offset += size(x,1)
	end
	o
end

function ChainRulesCore.ProjectTo(B::LazyVCatMatrix{T,N,Matrix{T}}) where {T,N}
	sizes = map(size, B.xs)
	element = eltype(B)
	return(ProjectTo{LazyVCatMatrix{T,N,Matrix{T}}}(;sizes, element))
end

function (a::ChainRulesCore.ProjectTo{LazyVCatMatrix{T,N,Matrix{T}}})(x::Matrix) where {T,N}
	offsets = cumsum((0, map(first, a.sizes)...))
	xs = map((i, j) -> x[i+1:j,:],offsets[1:end-1], offsets[2:end])
	return(Tangent{LazyVCatMatrix{T,N,Matrix{T}}}(;xs))
end


