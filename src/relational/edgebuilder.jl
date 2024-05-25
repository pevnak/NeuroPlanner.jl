"""
    EdgeBuilder

    Simplifies construction of edges in the graph. The graph is assumed to be undirected.
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
push!(eb, (2, 3))   # add edge (2, 3)

ds = construct(eb, :x)  # returns the computational graph pointing to `:x` in knowledge graph
BagNode  # 7 obs, 304 bytes
  ╰── ProductNode  # 2 obs, 32 bytes
        ├── ArrayNode(Colon()×2 KBEntry with Float32 elements)  # 2 obs, 88 bytes
        ╰── ArrayNode(Colon()×2 KBEntry with Float32 elements)  # 2 obs, 88 bytes
```    
"""
mutable struct EdgeBuilder{T<:Integer}
    indices::Matrix{T}
    num_vertices::Int
    arity::Int
    max_edges::Int
    num_edges::Int
    function EdgeBuilder(indices::Matrix{T}, num_vertices::Int, arity::Int, max_edges::Int) where {T}
        arity > 0 || error("Can create only edge of positive arity.")
        length(indices) == arity * max_edges || error("Length of indices must be equal to `arity * max_edges`.")
        num_edges = 0
        new{T}(indices, num_vertices, arity, max_edges, num_edges)
    end
end

function EdgeBuilder(::Val{arity}, max_edges::Int, num_vertices::Int) where {arity}
    indices = Matrix{Int}(undef, arity, max_edges)
    EdgeBuilder(indices, num_vertices, arity, max_edges)
end


function EdgeBuilder(arity::Int, max_edges::Int, num_vertices::Int)
    indices = Matrix{Int}(undef, arity, max_edges)
    EdgeBuilder(indices, num_vertices, arity, max_edges)
end

function Base.push!(eb::EdgeBuilder, vertices::NTuple{N,I}) where {N,I<:Integer}
    @assert all(v <= eb.num_vertices for v in vertices) "Cannot push edge connected to non-existant vertex!"
    @assert eb.arity == N "Cannot push edge of different arity to fixed size arity edge builder!"

    eb.num_edges += 1
    _mapenumerate_tuple(vertices) do i, vᵢ
        eb.indices[i, eb.num_edges] = vᵢ
    end
end

function construct(eb::EdgeBuilder, input_sym::Symbol)
    indices = view(eb.indices, :, 1:eb.num_edges)
    xs = Tuple([ArrayNode(KBEntry(input_sym, indices[i, :])) for i in 1:eb.arity])
    CompressedBagNode(ProductNode(xs), CompressedBags(indices, eb.num_vertices, eb.num_edges, eb.arity))
end


"""
    struct FeaturedEdgeBuilder{EB<:EdgeBuilder,M,F}
        eb::EB
        xe::M
        agg::F
    end


    Wraps `EdgeBuilder` and adds features on edges. The features on edges are mainly
    used to identify type of edge, as we want to compare between edges with features
    and multi-edges. When building the featured edges, there is also an option to 
    duplicate and deduplicate edges. When `agg = nothing,` edges are not deduplicated,
    which is faster during construction but it might be slower during inference. Conrary,
    when `agg=Function` (`+` is the default), the same edges are deduplicated and their 
    features are aggregated by `add`.
"""
struct FeaturedEdgeBuilder{EB<:EdgeBuilder,M,F}
    eb::EB
    xe::M
    agg::F
end

function FeaturedEdgeBuilder(arity::Int, max_edges::Int, num_vertices::Int, num_features::Int; agg=+)
    xe = zeros(Float32, num_features, max_edges)
    eb = EdgeBuilder(arity, max_edges, num_vertices)
    FeaturedEdgeBuilder(eb, xe, agg)
end

"""
    FeaturedEdgeBuilderNA(arity::Int, max_edges::Int, num_vertices::Int, num_features::Int)

    construct a `FeaturedEdgeBuilder` without aggregation
"""
function FeaturedEdgeBuilderNA(arity::Int, max_edges::Int, num_vertices::Int, num_features::Int)
    FeaturedEdgeBuilder(arity, max_edges, num_vertices, num_features;agg=nothing)    
end

function Base.push!(feb::FeaturedEdgeBuilder, vertices::NTuple{N,I}, x) where {N,I<:Integer}
    push!(feb.eb, vertices)
    feb.xe[:, feb.eb.num_edges] .= x
end

function Base.push!(feb::FeaturedEdgeBuilder, vertices::NTuple{N,I}, edge_type::Integer) where {N,I<:Integer}
    push!(feb.eb, vertices)
    feb.xe[edge_type, feb.eb.num_edges] = 1
end

"""
    construct(feb::FeaturedEdgeBuilder, input_sym::Symbol)

    A version of FeaturedEdgeBuilder, where edges are deduplicated and their features are
    aggregated by `feb.agg`
"""
function construct(feb::FeaturedEdgeBuilder{<:Any,<:Any,<:Function}, input_sym::Symbol)
    x = @view feb.xe[:, 1:feb.eb.num_edges]
    indices = @view feb.eb.indices[:, 1:feb.eb.num_edges]

    mask, ii = find_duplicates(indices)
    new_x = NNlib.scatter(feb.agg, x, ii)
    xs = Tuple([ArrayNode(KBEntry(input_sym, indices[i, mask])) for i in 1:feb.eb.arity])
    xs = tuple(xs..., ArrayNode(new_x))

    CompressedBagNode(ProductNode(xs), CompressedBags(indices[:, mask], feb.eb.num_vertices, sum(mask), feb.eb.arity))
end

"""
    construct(feb::FeaturedEdgeBuilder{<:Any,<:Any,Nothing}, input_sym::Symbol)

    if `agg` is nothing, then edges are not deduplicated
"""
function construct(feb::FeaturedEdgeBuilder{<:Any,<:Any,Nothing}, input_sym::Symbol)
    x = feb.xe[:, 1:feb.eb.num_edges]
    indices = @view feb.eb.indices[:, 1:feb.eb.num_edges]

    xs = Tuple([ArrayNode(KBEntry(input_sym, indices[i, :])) for i in 1:feb.eb.arity])
    xs = tuple(xs..., ArrayNode(x))
    CompressedBagNode(ProductNode(xs), CompressedBags(indices, feb.eb.num_vertices, feb.eb.num_edges, feb.eb.arity))
end


"""
    struct MultiEdgeBuilder{EB<:EdgeBuilder,T}
        eb::EB
        xe::Matrix{T}
    end

    Wraps `EdgeBuilder` and adds features on edges. When edges are constructed
    edges are duplicated and their feaures are summed. The `Matrix` for features
    one edges are duplicated. 
"""
struct MultiEdgeBuilder{N, EBS<:NTuple{N,<:EdgeBuilder}}
    ebs::EBS
end

function MultiEdgeBuilder(arity::Int, max_edges::Int, num_vertices::Int, num_features)
    ebs = tuple([EdgeBuilder(arity, max_edges, num_vertices) for _ in 1:num_features]...)
    MultiEdgeBuilder(ebs)
end

function Base.push!(feb::MultiEdgeBuilder, vertices::NTuple{N,I}, edge_type::Integer) where {N,I<:Integer}
    push!(feb.ebs[edge_type], vertices)
end

function construct(feb::MultiEdgeBuilder, input_sym::Symbol)
    ProductNode(map(Base.Fix2(construct, input_sym), feb.ebs))
end
