"""
    EdgeBuilder

    Simplifies construction of edges in the graph. The graph is assumed to be directed.
    The idea is that at the start, we know the maximal number of edges and vertices in the graph.
    The constructor allocates approximate space. Then we can add edges one by one using the `push!.`
    Finally, when done, we `construct`, returns the computational graph.


Example of use
```julia
nv = 7          # number of vertices in the graph
capacity = 5    # number of edges
arity = 2       # arity of the edge

eb = EdgeBuilder(arity, capacity, nv)
push!(eb, (1, 2))   # add edge (1, 2)
push!(eb, (2, 3))   # add edge (1, 2)

ds = construct(eb, :x)  # returns the computational graph pointing to `:x` in knowledge graph
BagNode  # 7 obs, 304 bytes
  ╰── ProductNode  # 2 obs, 32 bytes
        ├── ArrayNode(Colon()×2 KBEntry with Float32 elements)  # 2 obs, 88 bytes
        ╰── ArrayNode(Colon()×2 KBEntry with Float32 elements)  # 2 obs, 88 bytes
```    
"""
mutable struct EdgeBuilder{T<:Integer,I<:Integer}
    indices::Matrix{T}
    counts::Vector{I}
    arity::Int
    max_edges::Int
    offset::Int
    function EdgeBuilder(indices::Matrix{T}, counts::Vector{I}, arity::Int, max_edges::Int, offset::Int) where {T,I}
        arity > 0 || error("Can create only edge of positive arity.")
        offset == 0 || error("Offset needs to start at 0.")
        length(indices) == arity * max_edges || error("Length of indices must be equal to `arity * max_edges`.")
        new{T,I}(indices, counts, arity, max_edges, offset)
    end
end

function EdgeBuilder(::Val{arity}, max_edges::Int, num_vertices::Int) where {arity}
    indices = Matrix{Int}(undef, arity, max_edges)
    counts = fill(0, num_vertices)
    offset = 0
    EdgeBuilder(indices, counts, arity, max_edges, offset)
end


function EdgeBuilder(arity::Int, max_edges::Int, num_vertices::Int)
    indices = Matrix{Int}(undef, arity, max_edges)
    counts = fill(0, num_vertices)
    offset = 0
    EdgeBuilder(indices, counts, arity, max_edges, offset)
end

function Base.push!(eb::EdgeBuilder, vertices::NTuple{N,I}) where {N,I<:Integer}
    eb.offset += 1
    _mapenumerate_tuple(vertices) do i, vᵢ
        eb.counts[vᵢ] += 1
        eb.indices[i, eb.offset] = vᵢ
    end
end

function construct(eb::EdgeBuilder, input_sym::Symbol)
    xs = Tuple([ArrayNode(KBEntry(input_sym, eb.indices[i, 1:eb.offset])) for i in 1:eb.arity])
    CompressedBagNode(ProductNode(xs), CompressedBags(eb.indices, eb.counts, eb.max_edges, eb.offset))
end


